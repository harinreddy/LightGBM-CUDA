#ifdef USE_CUDA
#include "cuda_tree_learner.h"
#include "../io/dense_bin.hpp"
#include "../io/dense_nbits_bin.hpp"

#include <nvToolsExt.h>

#include <LightGBM/utils/array_args.h>
#include <LightGBM/network.h>
#include <LightGBM/bin.h>

#include <algorithm>
#include <vector>

#define cudaMemcpy_DEBUG 0 // 1: DEBUG cudaMemcpy
#define ResetTrainingData_DEBUG 0 // 1: Debug ResetTrainingData

#include <pthread.h>

//#define GPU_DEBUG_COMPARE // ibmGBT: For compare CPU and GPU histogram
#define GPU_DEBUG 0

static void *launch_cuda_histogram(void *thread_data) {
  ThreadData td = *(ThreadData*)thread_data;
  int device_id = td.device_id;
  CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));

  // launch cuda kernel
  cuda_histogram(td.leaf_num_data, td.num_data, td.use_all_features,
                td.is_constant_hessian, td.num_workgroups, td.stream,
                td.device_features,
                td.device_feature_masks,
                td.num_data,
                reinterpret_cast<uint*>(td.device_data_indices),
                td.leaf_num_data,
                td.device_gradients,
                td.device_hessians, td.hessians_const,
                td.device_subhistograms, td.sync_counters,
                td.device_histogram_outputs,
                td.exp_workgroups_per_feature);

  CUDASUCCESS_OR_FATAL(cudaGetLastError());

  return NULL;
}

static void *wait_event(void *wait_obj) {
  CUDASUCCESS_OR_FATAL(cudaEventSynchronize(*(cudaEvent_t *)wait_obj));
}

namespace LightGBM {

CUDATreeLearner::CUDATreeLearner(const Config* config)
  :SerialTreeLearner(config) {
  use_bagging_ = false;
  nthreads_ = 0;
  if(config->gpu_use_dp && USE_DP_FLOAT) Log::Info("This is the CUDA trainer with DP float!!");
  else Log::Info("This is the CUDA trainer with SP float!!");
}

CUDATreeLearner::~CUDATreeLearner() {

  // ibmGBT
  #ifdef TIMETAG
  for(int device_id = 0; device_id < num_gpu_; ++device_id) {
    Log::Info("CUDATreeLearner::Device[%d] kernel costs %f", device_id, kernel_time_[device_id] * 1e-3);
    Log::Info("CUDATreeLearner::Device[%d] kernel_input_memcopy costs %f", device_id, kernel_input_wait_time_[device_id] * 1e-3);
    Log::Info("CUDATreeLearner::Device[%d] kernel_output_memcopy costs %f", device_id, kernel_output_wait_time_[device_id] * 1e-3);
  }
  #endif
}


void CUDATreeLearner::Init(const Dataset* train_data, bool is_constant_hessian, bool is_use_subset) {

  // initialize SerialTreeLearner
  SerialTreeLearner::Init(train_data, is_constant_hessian, is_use_subset);

  // some additional variables needed for GPU trainer
  num_feature_groups_ = train_data_->num_feature_groups();
 
  // ibmGBT: use subset of training data for bagging
  is_use_subset_ = is_use_subset;  

  // Initialize GPU buffers and kernels & ibmGBT: get device info
  InitGPU(config_->num_gpu); // ibmGBT

}

// some functions used for debugging the GPU histogram construction

#ifdef GPU_DEBUG_COMPARE

void PrintHistograms(HistogramBinEntry* h, size_t size) {
  size_t total = 0;
  for (size_t i = 0; i < size; ++i) {
    printf("%03lu=%9.3g,%9.3g,%7d\t", i, h[i].sum_gradients, h[i].sum_hessians, h[i].cnt);
    total += h[i].cnt;
    if ((i & 3) == 3)
        printf("\n");
  }
  printf("\nTotal examples: %lu\n", total);
}

union Float_t
{
    int64_t i;
    double f;
    static int64_t ulp_diff(Float_t a, Float_t b) {
      return abs(a.i - b.i);
    }
};
  

void CompareHistograms(HistogramBinEntry* h1, HistogramBinEntry* h2, size_t size, int feature_id) {

  size_t i;
  Float_t a, b;
  for (i = 0; i < size; ++i) {
    a.f = h1[i].sum_gradients;
    b.f = h2[i].sum_gradients;
    int32_t ulps = Float_t::ulp_diff(a, b);
    if (fabs(h1[i].cnt           - h2[i].cnt != 0)) {
      printf("idx: %lu, %d != %d, (diff: %d, err_rate: %f)\n", i, h1[i].cnt, h2[i].cnt, h1[i].cnt - h2[i].cnt, (float)(h1[i].cnt - h2[i].cnt)/h2[i].cnt);
      //goto err;
    } else {
      //printf("idx: %lu, %d == %d\n", i, h1[i].cnt, h2[i].cnt);
      //printf("idx: %lu, pass\n", i);
    }
    if (ulps > 0) {
      printf("idx: %ld, grad %g != %g\n", i, h1[i].sum_gradients, h2[i].sum_gradients);
      //printf("idx: %ld, grad %g != %g (%d ULPs)\n", i, h1[i].sum_gradients, h2[i].sum_gradients, ulps);
      //goto err;
    }
    a.f = h1[i].sum_hessians;
    b.f = h2[i].sum_hessians;
    ulps = Float_t::ulp_diff(a, b);
    if (ulps > 0) {
      printf("idx: %ld, hessian %g != %g\n", i, h1[i].sum_hessians, h2[i].sum_hessians);
      //printf("idx: %ld, hessian %g != %g (%d ULPs)\n", i, h1[i].sum_hessians, h2[i].sum_hessians, ulps);
      // goto err;
    }
  }
  return;
err:
  Log::Warning("Mismatched histograms found for feature %d at location %lu.", feature_id, i);
}
#endif


int CUDATreeLearner::GetNumWorkgroupsPerFeature(data_size_t leaf_num_data) {

  // we roughly want 256 workgroups per device, and we have num_dense_feature4_ feature tuples.
  // also guarantee that there are at least 2K examples per workgroup

  double x = 256.0 / num_dense_feature_groups_;

  int exp_workgroups_per_feature = (int)ceil(log2(x));
  double t = leaf_num_data / 1024.0;

  #if GPU_DEBUG >= 4
  printf("Computing histogram for %d examples and (%d * %d) feature groups\n", leaf_num_data, dword_features_, num_dense_feature_groups_);
  printf("We can have at most %d workgroups per feature4 for efficiency reasons.\n"
         "Best workgroup size per feature for full utilization is %d\n", (int)ceil(t), (1 << exp_workgroups_per_feature));
  #endif

  exp_workgroups_per_feature = std::min(exp_workgroups_per_feature, (int)ceil(log((double)t)/log(2.0)));
  if (exp_workgroups_per_feature < 0)
      exp_workgroups_per_feature = 0;
  if (exp_workgroups_per_feature > kMaxLogWorkgroupsPerFeature)
      exp_workgroups_per_feature = kMaxLogWorkgroupsPerFeature;

  //printf("exp_workgroups_per_feature: %d\n", exp_workgroups_per_feature); // ibmGBT

  return exp_workgroups_per_feature;
}

void CUDATreeLearner::GPUHistogram(data_size_t leaf_num_data, bool use_all_features) {
  nvtxRangePush(__PRETTY_FUNCTION__);

  // we have already copied ordered gradients, ordered hessians and indices to GPU
  // decide the best number of workgroups working on one feature4 tuple
  // set work group size based on feature size
  // each 2^exp_workgroups_per_feature workgroups work on a feature4 tuple


  int exp_workgroups_per_feature = GetNumWorkgroupsPerFeature(leaf_num_data);
  std::vector<int> num_gpu_workgroups;
  ThreadData *thread_data = (ThreadData*)malloc(sizeof(ThreadData) * num_gpu_);

  for(int device_id = 0; device_id < num_gpu_; ++device_id) {
    int num_gpu_feature_groups = num_gpu_feature_groups_[device_id];
    int num_workgroups = (1 << exp_workgroups_per_feature) * num_gpu_feature_groups;
    num_gpu_workgroups.push_back(num_workgroups);
    if (num_workgroups > preallocd_max_num_wg_[device_id]) {
      Log::Debug("num_workgroups > preallocd_max_num_wg_");
      preallocd_max_num_wg_.at(device_id) = num_workgroups;
      Log::Debug("Increasing preallocd_max_num_wg_ to %d for launching more workgroups", preallocd_max_num_wg_[device_id]);
      CUDASUCCESS_OR_FATAL(cudaFree(device_subhistograms_[device_id]));
      cudaMalloc(&(device_subhistograms_[device_id]), (size_t) num_workgroups * dword_features_ * device_bin_size_ * hist_bin_entry_sz_);
    }
    //set thread_data
    SetThreadData(thread_data, device_id, leaf_num_data, use_all_features,
                  num_workgroups, exp_workgroups_per_feature);
  }
  
  for(int device_id = 0; device_id < num_gpu_; ++device_id) {
    if (pthread_create(cpu_threads_[device_id], NULL, launch_cuda_histogram, (void *)(&thread_data[device_id]))){
        fprintf(stderr, "Error in creating threads. Exiting\n");
        exit(0);
    }
  }

  /* Wait for the threads to finish */

  for(int device_id = 0; device_id < num_gpu_; ++device_id) {
    if (pthread_join(*(cpu_threads_[device_id]), NULL)){
      fprintf(stderr, "Error in joining threads. Exiting\n");
      exit(0);
    }
  }

  for(int device_id = 0; device_id < num_gpu_; ++device_id) {

    // copy the results asynchronously. Size depends on if double precision is used

    size_t output_size = num_gpu_feature_groups_[device_id] * dword_features_ * device_bin_size_ * hist_bin_entry_sz_;
    size_t host_output_offset = offset_gpu_feature_groups_[device_id] * dword_features_ * device_bin_size_ * hist_bin_entry_sz_;

    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync((char*)host_histogram_outputs_ + host_output_offset, device_histogram_outputs_[device_id], output_size, cudaMemcpyDeviceToHost, stream_[device_id]));

    Log::Debug("Enqueued cudaMemcpyAsync of device_histogram_outputs_ -> host_histogram_outputs_");

    CUDASUCCESS_OR_FATAL(cudaEventRecord(histograms_wait_obj_[device_id], stream_[device_id]));
  }

  Log::Debug("Returning from GPUHistogram");
  nvtxRangePop();
}


template <typename HistType>
void CUDATreeLearner::WaitAndGetHistograms(HistogramBinEntry* histograms) {
  nvtxRangePush(__PRETTY_FUNCTION__);
  HistType* hist_outputs = (HistType*) host_histogram_outputs_;

  //#pragma omp parallel for schedule(static, num_gpu_)
  for(int device_id = 0; device_id < num_gpu_; ++device_id) {

    #ifdef TIMETAG
    auto start_time = std::chrono::steady_clock::now();
    #endif

    // when the output is ready, the computation is done
    CUDASUCCESS_OR_FATAL(cudaEventSynchronize(histograms_wait_obj_[device_id]));
    // ibmGBT

    #ifdef TIMETAG
    auto kernel_output_wait_time = std::chrono::steady_clock::now() - start_time;
    kernel_output_wait_time_.at(device_id) += kernel_output_wait_time;
    #endif
  }

  #pragma omp parallel for schedule(static)
  for(int i = 0; i < num_dense_feature_groups_; ++i) {
    if (!feature_masks_[i]) {
      continue;
    }
    int dense_group_index = dense_feature_group_map_[i];
    auto old_histogram_array = histograms + train_data_->GroupBinBoundary(dense_group_index);
    int bin_size = train_data_->FeatureGroupNumBin(dense_group_index); 
    for (int j = 0; j < bin_size; ++j) {
      old_histogram_array[j].sum_gradients = hist_outputs[i * device_bin_size_+ j].sum_gradients;
      old_histogram_array[j].sum_hessians = hist_outputs[i * device_bin_size_ + j].sum_hessians;
      old_histogram_array[j].cnt = (data_size_t)hist_outputs[i * device_bin_size_ + j].cnt;
    }
  }

  nvtxRangePop();
}

// ibmGBT
void CUDATreeLearner::CountDenseFeatureGroups() {
  nvtxRangePush(__PRETTY_FUNCTION__);

  num_dense_feature_groups_ = 0;

  for (int i = 0; i < num_feature_groups_; ++i) {
    if (ordered_bins_[i] == nullptr) {
      num_dense_feature_groups_++;
    }
  }
  if (!num_dense_feature_groups_) {
    Log::Warning("GPU acceleration is disabled because no non-trival dense features can be found");
  }

  nvtxRangePop();
}

// ibmGBT
void CUDATreeLearner::prevAllocateGPUMemory() {

  nvtxRangePush(__PRETTY_FUNCTION__);

  // how many feature-group tuples we have
  // leave some safe margin for prefetching
  // 256 work-items per workgroup. Each work-item prefetches one tuple for that feature

  allocated_num_data_ = num_data_ + 256 * (1 << kMaxLogWorkgroupsPerFeature);

  // clear sparse/dense maps

  dense_feature_group_map_.clear();
  sparse_feature_group_map_.clear();

  // do nothing it there is no dense feature
  if (!num_dense_feature_groups_) {
    return;
  }

  // ibmGBT: calculate number of feature groups per gpu
  num_gpu_feature_groups_.resize(num_gpu_);
  offset_gpu_feature_groups_.resize(num_gpu_);
  int num_features_per_gpu = num_dense_feature_groups_ / num_gpu_;
  int remain_features = num_dense_feature_groups_ - num_features_per_gpu * num_gpu_;

  int offset = 0;

  for(int i = 0; i < num_gpu_; ++i) {
    offset_gpu_feature_groups_.at(i) = offset;
    num_gpu_feature_groups_.at(i) = (i < remain_features)? num_features_per_gpu + 1 : num_features_per_gpu;
    offset += num_gpu_feature_groups_.at(i);
  }

  // allocate feature mask, for disabling some feature-groups' histogram calculation
  if (feature_masks_.data() != NULL) {
     cudaPointerAttributes attributes;
     cudaPointerGetAttributes (&attributes, feature_masks_.data());
    
     if ((attributes.type == cudaMemoryTypeHost) && (attributes.devicePointer != NULL)){ 
        CUDASUCCESS_OR_FATAL(cudaHostUnregister(feature_masks_.data()));
     }
  }
  feature_masks_.resize(num_dense_feature_groups_);
  Log::Debug("Resized feature masks");

  ptr_pinned_feature_masks_ = feature_masks_.data();
  Log::Debug("Memset pinned_feature_masks_");
  memset(ptr_pinned_feature_masks_, 0, num_dense_feature_groups_);

  // histogram bin entry size depends on the precision (single/double)
  hist_bin_entry_sz_ = config_->gpu_use_dp ? sizeof(HistogramBinEntry) : sizeof(GPUHistogramBinEntry);

  // host_size histogram outputs
  //  host_histogram_outputs_ = malloc(num_dense_feature_groups_ * device_bin_size_ * hist_bin_entry_sz_);

  CUDASUCCESS_OR_FATAL(cudaHostAlloc( (void **)&host_histogram_outputs_, (size_t)(num_dense_feature_groups_ * device_bin_size_ * hist_bin_entry_sz_),cudaHostAllocPortable));

  // ibmGBT
  nthreads_ = std::min(omp_get_max_threads(), num_dense_feature_groups_ / dword_features_);
  nthreads_ = std::max(nthreads_, 1);
  nvtxRangePop();
}

// ibmGBT: allocate GPU memory for each GPU
void CUDATreeLearner::AllocateGPUMemory() {

  nvtxRangePush(__PRETTY_FUNCTION__);

  #pragma omp parallel for schedule(static, num_gpu_)

  for(int device_id = 0; device_id < num_gpu_; ++device_id) {
    // do nothing it there is no gpu feature
    int num_gpu_feature_groups = num_gpu_feature_groups_[device_id];
    if (num_gpu_feature_groups) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));

      // allocate memory for all features (FIXME: 4 GB barrier on some devices, need to split to multiple buffers)
      if ( device_features_[device_id] != NULL ) {
             CUDASUCCESS_OR_FATAL(cudaFree(device_features_[device_id]));
      }

      Log::Debug("Pre Allocated device_features_ addr=%p sz=%lu", device_features_[device_id], num_gpu_feature_groups * num_data_);
      CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_features_[device_id]),  (size_t)num_gpu_feature_groups * num_data_ * sizeof(uint8_t)));
      Log::Debug("Allocated device_features_ addr=%p sz=%lu", device_features_[device_id], num_gpu_feature_groups * num_data_);

      // allocate space for gradients and hessians on device
      // we will copy gradients and hessians in after ordered_gradients_ and ordered_hessians_ are constructed

      if (device_gradients_[device_id] != NULL){
        CUDASUCCESS_OR_FATAL(cudaFree(device_gradients_[device_id]));
      }

      if (device_hessians_[device_id] != NULL){
        CUDASUCCESS_OR_FATAL(cudaFree(device_hessians_[device_id]));
      }

      if (device_feature_masks_[device_id] != NULL){
         CUDASUCCESS_OR_FATAL(cudaFree(device_feature_masks_[device_id]));
      }

      CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_gradients_[device_id]), (size_t) allocated_num_data_ * sizeof(score_t)));
      CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_hessians_[device_id]),  (size_t) allocated_num_data_ * sizeof(score_t)));

      CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_feature_masks_[device_id]), (size_t) num_gpu_feature_groups));

      // copy indices to the device

     if (device_feature_masks_[device_id] != NULL){
        CUDASUCCESS_OR_FATAL(cudaFree(device_data_indices_[device_id])); 
     }

      CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_data_indices_[device_id]), (size_t) allocated_num_data_ * sizeof(data_size_t)));
      CUDASUCCESS_OR_FATAL(cudaMemsetAsync(device_data_indices_[device_id], 0, allocated_num_data_ * sizeof(data_size_t), stream_[device_id]));

      Log::Debug("Memset device_data_indices_");

      // create output buffer, each feature has a histogram with device_bin_size_ bins,
      // each work group generates a sub-histogram of dword_features_ features.

      if (!device_subhistograms_[device_id]) {

  // only initialize once here, as this will not need to change when ResetTrainingData() is called
        CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_subhistograms_[device_id]), (size_t) preallocd_max_num_wg_[device_id] * dword_features_ * device_bin_size_ * hist_bin_entry_sz_));

        Log::Debug("created device_subhistograms_: %p", device_subhistograms_[device_id]);

      }

      // create atomic counters for inter-group coordination
      CUDASUCCESS_OR_FATAL(cudaFree(sync_counters_[device_id]));
      CUDASUCCESS_OR_FATAL(cudaMalloc(&(sync_counters_[device_id]), (size_t) num_gpu_feature_groups * sizeof(int))); 
      CUDASUCCESS_OR_FATAL(cudaMemsetAsync(sync_counters_[device_id], 0, num_gpu_feature_groups * sizeof(int)));

      // The output buffer is allocated to host directly, to overlap compute and data transfer
      CUDASUCCESS_OR_FATAL(cudaFree(device_histogram_outputs_[device_id]));
      CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_histogram_outputs_[device_id]), (size_t) num_gpu_feature_groups * device_bin_size_ * hist_bin_entry_sz_));
    }
  }

  nvtxRangePop();
}

void CUDATreeLearner::ResetGPUMemory() {
  nvtxRangePush(__PRETTY_FUNCTION__);

  // clear sparse/dense maps
  dense_feature_group_map_.clear();
  sparse_feature_group_map_.clear();

  nvtxRangePop();
}

// ibmGBT
void CUDATreeLearner::copyDenseFeature() {
  nvtxRangePush(__PRETTY_FUNCTION__);
  auto start_time = std::chrono::steady_clock::now();
  Log::Debug("Started copying dense features from CPU to GPU");
  // find the dense feature-groups and group then into Feature4 data structure (several feature-groups packed into 4 bytes)
  size_t  copied_feature = 0;
  // set device info 
  int device_id = 0;
  uint8_t* device_features = device_features_[device_id];
  Log::Debug("Started copying dense features from CPU to GPU - 1");

  for (int i = 0; i < num_feature_groups_; ++i) {
    // looking for dword_features_ non-sparse feature-groups
    if (ordered_bins_[i] == nullptr) {
      dense_feature_group_map_.push_back(i);
      auto sizes_in_byte = train_data_->FeatureGroupSizesInByte(i);
      void* tmp_data = train_data_->FeatureGroupData(i);
  	   Log::Debug("Started copying dense features from CPU to GPU - 2");
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(&device_features[copied_feature * num_data_], tmp_data, sizes_in_byte, cudaMemcpyHostToDevice, stream_[device_id]));
  	   Log::Debug("Started copying dense features from CPU to GPU - 3");
      copied_feature++;
      // reset device info
      if(copied_feature == num_gpu_feature_groups_[device_id]) {
         CUDASUCCESS_OR_FATAL(cudaEventRecord(features_future_[device_id], stream_[device_id]));
         device_id += 1;
         copied_feature = 0;
         if(device_id < num_gpu_) {
           device_features = device_features_[device_id];
           //CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id)); 
         }
      }
    }
    else {
      sparse_feature_group_map_.push_back(i);
    }
  }


  // data transfer time // ibmGBT: async copy, so it is not the real data transfer time
  std::chrono::duration<double, std::milli> end_time = std::chrono::steady_clock::now() - start_time;

  nvtxRangePop();
}



// ibmGBT: InitGPU w/ num_gpu
void CUDATreeLearner::InitGPU(int num_gpu) { 

  // Get the max bin size, used for selecting best GPU kernel

  max_num_bin_ = 0;

  #if GPU_DEBUG >= 1
  printf("bin size: ");
  #endif
  for (int i = 0; i < num_feature_groups_; ++i) {
    #if GPU_DEBUG >= 1
    printf("%d, ", train_data_->FeatureGroupNumBin(i));
    #endif
    max_num_bin_ = std::max(max_num_bin_, train_data_->FeatureGroupNumBin(i));
  }

  if (max_num_bin_ <= 16) {
    device_bin_size_ = 256; //ibmGBT
    dword_features_ = 1; // ibmGBT
  }
  else if (max_num_bin_ <= 64) {
    device_bin_size_ = 256; //ibmGBT
    dword_features_ = 1; // ibmGBT
  }
  else if ( max_num_bin_ <= 256) {
    Log::Debug("device_bin_size_ = 256");
    device_bin_size_ = 256;
    dword_features_ = 1; // ibmGBT
  }
  else {
    Log::Fatal("bin size %d cannot run on GPU", max_num_bin_);
  }
  if(max_num_bin_ == 65) {
    Log::Warning("Setting max_bin to 63 is sugguested for best performance");
  }
  if(max_num_bin_ == 17) {
    Log::Warning("Setting max_bin to 15 is sugguested for best performance");
  }

  // ibmGBT: get num_dense_feature_groups_
  CountDenseFeatureGroups();

  if (num_gpu > num_dense_feature_groups_) num_gpu = num_dense_feature_groups_;
 
  // ibmGBT: initialize GPU
  int gpu_count;

  CUDASUCCESS_OR_FATAL(cudaGetDeviceCount(&gpu_count));
  num_gpu_ = (gpu_count < num_gpu)? gpu_count : num_gpu;

  // ibmGBT: set cpu threads
  cpu_threads_ = (pthread_t **)malloc(sizeof(pthread_t *)*num_gpu_);
  for(int device_id = 0; device_id < num_gpu_; ++device_id) {
    cpu_threads_[device_id] = (pthread_t *)malloc(sizeof(pthread_t)); 
  }

  // ibmGBT: resize device memory pointers
  device_features_.resize(num_gpu_);
  device_gradients_.resize(num_gpu_);
  device_hessians_.resize(num_gpu_);
  device_feature_masks_.resize(num_gpu_);
  device_data_indices_.resize(num_gpu_);
  sync_counters_.resize(num_gpu_);
  device_subhistograms_.resize(num_gpu_);
  device_histogram_outputs_.resize(num_gpu_);
 
  // ibmGBT: create stream & events to handle multiple GPUs
  preallocd_max_num_wg_.resize(num_gpu_, 1024);
  stream_.resize(num_gpu_);
  hessians_future_.resize(num_gpu_);
  gradients_future_.resize(num_gpu_);
  indices_future_.resize(num_gpu_);
  features_future_.resize(num_gpu_);
  kernel_start_.resize(num_gpu_);
  kernel_wait_obj_.resize(num_gpu_);
  histograms_wait_obj_.resize(num_gpu_);

  // for debuging
  kernel_time_.resize(num_gpu_, 0);
  kernel_input_wait_time_.resize(num_gpu_, std::chrono::milliseconds(0));
  kernel_output_wait_time_.resize(num_gpu_, std::chrono::milliseconds(0));


  for(int i = 0; i < num_gpu_; ++i) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(i));
    CUDASUCCESS_OR_FATAL(cudaStreamCreate(&(stream_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(hessians_future_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(gradients_future_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(indices_future_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(features_future_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(kernel_start_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(kernel_wait_obj_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(histograms_wait_obj_[i])));
  }

  prevAllocateGPUMemory();

  AllocateGPUMemory();

  // ibmGBT: copy dense feature data from cpu to gpu only when we use entire traning data for training

  if (!is_use_subset_) {
    Log::Debug("copyDenseFeature at the initialization\n");
    copyDenseFeature(); // ibmGBT
  }

}

Tree* CUDATreeLearner::Train(const score_t* gradients, const score_t *hessians,
                            bool is_constant_hessian, Json& forced_split_json) {
  nvtxRangePush(__PRETTY_FUNCTION__);

  // check if we need to recompile the GPU kernel (is_constant_hessian changed)
  // this should rarely occur

  if (is_constant_hessian != is_constant_hessian_) {
    Log::Debug("Recompiling GPU kernel because hessian is %sa constant now", is_constant_hessian ? "" : "not ");
    is_constant_hessian_ = is_constant_hessian;
  }

  Tree *ret = SerialTreeLearner::Train(gradients, hessians, is_constant_hessian, forced_split_json);
  nvtxRangePop();

  return ret;
}

void CUDATreeLearner::ResetTrainingData(const Dataset* train_data) {

  // ibmGBT: check data size
  data_size_t old_num_data = num_data_;  

  SerialTreeLearner::ResetTrainingData(train_data);

  #if ResetTrainingData_DEBUG == 1 // ibmGBT 
  serial_time = std::chrono::steady_clock::now() - start_serial_time;  
  #endif

  num_feature_groups_ = train_data_->num_feature_groups();

  // GPU memory has to been reallocated because data may have been changed

  #if ResetTrainingData_DEBUG == 1 // ibmGBT
  auto start_alloc_gpu_time = std::chrono::steady_clock::now();
  #endif

  // ibmGBT: AllocateGPUMemory only when the number of data increased

  int old_num_feature_groups = num_dense_feature_groups_;
  CountDenseFeatureGroups();
  if ((old_num_data < num_data_) & (old_num_feature_groups < num_dense_feature_groups_)) {
    prevAllocateGPUMemory();
    AllocateGPUMemory();
  } else {
    ResetGPUMemory();
  }

  copyDenseFeature();

  #if ResetTrainingData_DEBUG == 1 // ibmGBT
  alloc_gpu_time = std::chrono::steady_clock::now() - start_alloc_gpu_time;
  #endif

  // setup GPU kernel arguments after we allocating all the buffers

  #if ResetTrainingData_DEBUG == 1 // ibmGBT
  auto start_set_arg_time = std::chrono::steady_clock::now();
  #endif

  #if ResetTrainingData_DEBUG == 1 // ibmGBT
  set_arg_time = std::chrono::steady_clock::now() - start_set_arg_time;
  reset_training_data_time = std::chrono::steady_clock::now() - start_reset_training_data_time;
  Log::Info("reset_training_data_time: %f secs.", reset_training_data_time.count() * 1e-3);
  Log::Info("serial_time: %f secs.", serial_time.count() * 1e-3);
  Log::Info("alloc_gpu_time: %f secs.", alloc_gpu_time.count() * 1e-3);
  Log::Info("set_arg_time: %f secs.", set_arg_time.count() * 1e-3); 
  #endif
}

void CUDATreeLearner::BeforeTrain() {
  nvtxRangePush(__PRETTY_FUNCTION__);
  Log::Debug("CUDATreeLearner::BeforeTrain() entered...");

  #if cudaMemcpy_DEBUG == 1 // ibmGBT
  std::chrono::duration<double, std::milli> device_hessians_time = std::chrono::milliseconds(0);
  std::chrono::duration<double, std::milli> device_gradients_time = std::chrono::milliseconds(0);
  #endif

  #if GPU_DEBUG >= 2
  printf("Copying intial full gradients and hessians to device\n");
  #endif

  // Copy initial full hessians and gradients to GPU.
  // We start copying as early as possible, instead of at ConstructHistogram().

  if (!use_bagging_ && num_dense_feature_groups_) {

    Log::Debug("No baggings, dense_feature_groups_");

    for(int device_id = 0; device_id < num_gpu_; ++device_id) {
      if (!is_constant_hessian_) {
        Log::Debug("CUDATreeLearner::BeforeTrain(): Starting hessians_ -> device_hessians_");

        #if cudaMemcpy_DEBUG == 1  // ibmGBT
        auto start_device_hessians_time = std::chrono::steady_clock::now();
        #endif

    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_hessians_[device_id], hessians_, num_data_*sizeof(score_t), cudaMemcpyHostToDevice, stream_[device_id]));

        CUDASUCCESS_OR_FATAL(cudaEventRecord(hessians_future_[device_id], stream_[device_id]));

        #if cudaMemcpy_DEBUG == 1  // ibmGBT
        device_hessians_time = std::chrono::steady_clock::now() - start_device_hessians_time;
        #endif

        Log::Debug("queued copy of device_hessians_");
      }

      #if cudaMemcpy_DEBUG == 1  // ibmGBT
      auto start_device_gradients_time = std::chrono::steady_clock::now();
      #endif

      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_gradients_[device_id], gradients_, num_data_ * sizeof(score_t), cudaMemcpyHostToDevice, stream_[device_id]));
      CUDASUCCESS_OR_FATAL(cudaEventRecord(gradients_future_[device_id], stream_[device_id]));

      #if cudaMemcpy_DEBUG == 1  // ibmGBT
      device_gradients_time = std::chrono::steady_clock::now() - start_device_gradients_time;
      #endif

      Log::Debug("CUDATreeLearner::BeforeTrain: issued gradients_ -> device_gradients_");
   }
  }

  Log::Debug("SerialTreeLearner::BeforeTrain()");

  SerialTreeLearner::BeforeTrain();

  Log::Debug("SerialTreeLearner::BeforeTrain() done");

  // use bagging
  if (data_partition_->leaf_count(0) != num_data_ && num_dense_feature_groups_) {

    // On GPU, we start copying indices, gradients and hessians now, instead at ConstructHistogram()
    // copy used gradients and hessians to ordered buffer

    const data_size_t* indices = data_partition_->indices();
    data_size_t cnt = data_partition_->leaf_count(0);

    #if GPU_DEBUG > 0
    printf("Using bagging, examples count = %d\n", cnt);
    #endif

    // transfer the indices to GPU
    for(int device_id = 0; device_id < num_gpu_; ++device_id) {

      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_data_indices_[device_id], indices, cnt * sizeof(*indices), cudaMemcpyHostToDevice, stream_[device_id]));
      CUDASUCCESS_OR_FATAL(cudaEventRecord(indices_future_[device_id], stream_[device_id]));

      if (!is_constant_hessian_) {

        Log::Debug("CUDATreeLearner::BeforeTrain(): Starting hessians_ -> device_hessians_");
        CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_hessians_[device_id], hessians_, num_data_ * sizeof(score_t), cudaMemcpyHostToDevice, stream_[device_id]));
        CUDASUCCESS_OR_FATAL(cudaEventRecord(hessians_future_[device_id], stream_[device_id]));
        Log::Debug("queued copy of device_hessians_");

      }

      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_gradients_[device_id], gradients_, num_data_ * sizeof(score_t), cudaMemcpyHostToDevice, stream_[device_id]));
      CUDASUCCESS_OR_FATAL(cudaEventRecord(gradients_future_[device_id], stream_[device_id]));
    }

    Log::Debug("CUDATreeLearner::BeforeTrain: issued gradients_ -> device_gradients_");
  }

  Log::Debug("BeforeTrain() exiting...");
  nvtxRangePop();
}

bool CUDATreeLearner::BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) {
  nvtxRangePush(__PRETTY_FUNCTION__);

  int smaller_leaf;

  data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
  data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);

  // only have root
  if (right_leaf < 0) {
    smaller_leaf = -1;
  } else if (num_data_in_left_child < num_data_in_right_child) {
    smaller_leaf = left_leaf;
  } else {
    smaller_leaf = right_leaf;
  }

  // Copy indices, gradients and hessians as early as possible
  if (smaller_leaf >= 0 && num_dense_feature_groups_) {
    // only need to initialize for smaller leaf
    // Get leaf boundary
    const data_size_t* indices = data_partition_->indices();
    data_size_t begin = data_partition_->leaf_begin(smaller_leaf);
    data_size_t end = begin + data_partition_->leaf_count(smaller_leaf);

    // copy indices to the GPU:
    #if GPU_DEBUG >= 2
    Log::Info("Copying indices, gradients and hessians to GPU...");
    #endif

    for(int device_id = 0; device_id < num_gpu_; ++device_id) {
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_data_indices_[device_id], &indices[begin], (end-begin) * sizeof(data_size_t), cudaMemcpyHostToDevice, stream_[device_id]));
      CUDASUCCESS_OR_FATAL(cudaEventRecord(indices_future_[device_id], stream_[device_id]));
    }
  }

  const bool ret = SerialTreeLearner::BeforeFindBestSplit(tree, left_leaf, right_leaf);
  nvtxRangePop();

  return ret;
}

bool CUDATreeLearner::ConstructGPUHistogramsAsync(
  const std::vector<int8_t>& is_feature_used,
  const data_size_t* data_indices, data_size_t num_data) {

  nvtxRangePush(__PRETTY_FUNCTION__);
  Log::Debug("ConstructGPUHistogramsAsync entered...");

  if (num_data <= 0) {
    Log::Debug("num_data <0, returning");
    return false;
  }

  // do nothing if no features can be processed on GPU
  if (!num_dense_feature_groups_) {
    Log::Debug("no dense feature groups, returning");
    return false;
  }
  
  // copy data indices if it is not null
  if (data_indices != nullptr && num_data != num_data_) {
    Log::Debug("starting async data_indices -> device_data_indices");

    for(int device_id = 0; device_id < num_gpu_; ++device_id) {

      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_data_indices_[device_id], data_indices, num_data * sizeof(data_size_t), cudaMemcpyHostToDevice, stream_[device_id]));
      CUDASUCCESS_OR_FATAL(cudaEventRecord(indices_future_[device_id], stream_[device_id]));

    }
    Log::Debug("async data_indices -> device_data_indices");
  }

  // converted indices in is_feature_used to feature-group indices
  std::vector<int8_t> is_feature_group_used(num_feature_groups_, 0);

  #pragma omp parallel for schedule(static,1024) if (num_features_ >= 2048)
  for (int i = 0; i < num_features_; ++i) {
    if(is_feature_used[i]) { 
      int feature_group = train_data_->Feature2Group(i); // ibmGBT
      is_feature_group_used[feature_group] = (train_data_->FeatureGroupNumBin(feature_group)<=16)? 2 : 1; // ibmGBT
    }
  }

  // construct the feature masks for dense feature-groups
  int used_dense_feature_groups = 0;
  #pragma omp parallel for schedule(static,1024) reduction(+:used_dense_feature_groups) if (num_dense_feature_groups_ >= 2048)
  for (int i = 0; i < num_dense_feature_groups_; ++i) {
    if (is_feature_group_used[dense_feature_group_map_[i]]) {
      //feature_masks_[i] = 1;
      feature_masks_[i] = is_feature_group_used[dense_feature_group_map_[i]];
      ++used_dense_feature_groups;
    }
    else {
      feature_masks_[i] = 0;
    }
  }
  bool use_all_features = used_dense_feature_groups == num_dense_feature_groups_;
  // if no feature group is used, just return and do not use GPU
  if (used_dense_feature_groups == 0) {
    return false;
  }

#if GPU_DEBUG >= 1
  printf("Feature masks:\n");
  for (unsigned int i = 0; i < feature_masks_.size(); ++i) {
    printf("%d ", feature_masks_[i]);
  }
  printf("\n");
  printf("%d feature groups, %d used, %d\n", num_dense_feature_groups_, used_dense_feature_groups, use_all_features);
#endif

  // if not all feature groups are used, we need to transfer the feature mask to GPU
  // otherwise, we will use a specialized GPU kernel with all feature groups enabled
  // ibmGBT FIXME: No waiting mark for feature mask

  if (!use_all_features) {
    //#pragma omp parallel for schedule(static, num_gpu_)
    for(int device_id = 0; device_id < num_gpu_; ++device_id) {
      int offset = offset_gpu_feature_groups_[device_id];
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_feature_masks_[device_id], ptr_pinned_feature_masks_ + offset, num_gpu_feature_groups_[device_id] , cudaMemcpyHostToDevice, stream_[device_id]));
    }
  }

  // All data have been prepared, now run the GPU kernel
  Log::Debug("calling GPUHistogram");
  GPUHistogram(num_data, use_all_features);
  Log::Debug("returned from GPUHistogram to ConstructGPUHistogramsAsync");

  nvtxRangePop();
  return true;
}

void CUDATreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) {
  nvtxRangePush(__PRETTY_FUNCTION__);
  //ibmGBT
  #ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
  #endif

  std::vector<int8_t> is_sparse_feature_used(num_features_, 0);
  std::vector<int8_t> is_dense_feature_used(num_features_, 0);

  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {

    if (!is_feature_used_[feature_index]) continue;
    if (!is_feature_used[feature_index]) continue;
    if (ordered_bins_[train_data_->Feature2Group(feature_index)]) {
      is_sparse_feature_used[feature_index] = 1;
    }
    else {
      is_dense_feature_used[feature_index] = 1;
    }
  }

  // construct smaller leaf
  HistogramBinEntry* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - 1;
  // ConstructGPUHistogramsAsync will return true if there are availabe feature gourps dispatched to GPU
  bool is_gpu_used = ConstructGPUHistogramsAsync(is_feature_used,
    nullptr, smaller_leaf_splits_->num_data_in_leaf());
  // then construct sparse features on CPU
  // We set data_indices to null to avoid rebuilding ordered gradients/hessians
  train_data_->ConstructHistograms(is_sparse_feature_used,
    nullptr, smaller_leaf_splits_->num_data_in_leaf(),
    smaller_leaf_splits_->LeafIndex(),
    ordered_bins_, gradients_, hessians_,
    ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
    ptr_smaller_leaf_hist_data);
  // wait for GPU to finish, only if GPU is actually used
  if (is_gpu_used) {
    if (config_->gpu_use_dp) {
      // use double precision
      WaitAndGetHistograms<HistogramBinEntry>(ptr_smaller_leaf_hist_data);
    }
    else {
      // use single precision
      WaitAndGetHistograms<GPUHistogramBinEntry>(ptr_smaller_leaf_hist_data);
    }
  }

  // Compare GPU histogram with CPU histogram, useful for debuggin GPU code problem
  // #define GPU_DEBUG_COMPARE
  #ifdef GPU_DEBUG_COMPARE
  printf("Start Comparing Histogram between GPU and CPU\n");
  bool compare = true;
  for (int i = 0; i < num_dense_feature_groups_; ++i) {
    if (!feature_masks_[i])
      continue;
    int dense_feature_group_index = dense_feature_group_map_[i];
    size_t size = train_data_->FeatureGroupNumBin(dense_feature_group_index);
    HistogramBinEntry* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - 1;
    HistogramBinEntry* current_histogram = ptr_smaller_leaf_hist_data + train_data_->GroupBinBoundary(dense_feature_group_index);
    HistogramBinEntry* gpu_histogram = new HistogramBinEntry[size];
    data_size_t num_data = smaller_leaf_splits_->num_data_in_leaf();

    std::copy(current_histogram, current_histogram + size, gpu_histogram);
    std::memset(current_histogram, 0, train_data_->FeatureGroupNumBin(dense_feature_group_index) * sizeof(HistogramBinEntry));
    if ( num_data == num_data_ ) {
      if ( is_constant_hessian_ ) {
        train_data_->FeatureGroupBin(dense_feature_group_index)->ConstructHistogram(
            num_data,
            gradients_,
            current_histogram);
      } else {
        train_data_->FeatureGroupBin(dense_feature_group_index)->ConstructHistogram(
            num_data,
            gradients_, hessians_,
            current_histogram);
      }
    } else {
      if ( is_constant_hessian_ ) {
        train_data_->FeatureGroupBin(dense_feature_group_index)->ConstructHistogram(
            smaller_leaf_splits_->data_indices(),
            num_data,
            ordered_gradients_.data(),
            current_histogram);
      } else {
        train_data_->FeatureGroupBin(dense_feature_group_index)->ConstructHistogram(
            smaller_leaf_splits_->data_indices(),
            num_data,
            ordered_gradients_.data(), ordered_hessians_.data(),
            current_histogram);
      }
    }
    if ( (num_data != num_data_) & compare ) {
        CompareHistograms(gpu_histogram, current_histogram, size, dense_feature_group_index);
        compare = false;
    }
    //CompareHistograms(gpu_histogram, current_histogram, size, dense_feature_group_index);
    std::copy(gpu_histogram, gpu_histogram + size, current_histogram);
    delete [] gpu_histogram;
    //break; // ibmGBT: see only first feature info
  }
  printf("End Comparing Histogram between GPU and CPU\n");
  #endif

  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {

    // construct larger leaf

    HistogramBinEntry* ptr_larger_leaf_hist_data = larger_leaf_histogram_array_[0].RawData() - 1;

    is_gpu_used = ConstructGPUHistogramsAsync(is_feature_used,
      larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf());

    // then construct sparse features on CPU
    // We set data_indices to null to avoid rebuilding ordered gradients/hessians

    train_data_->ConstructHistograms(is_sparse_feature_used,
      nullptr, larger_leaf_splits_->num_data_in_leaf(),
      larger_leaf_splits_->LeafIndex(),
      ordered_bins_, gradients_, hessians_,
      ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
      ptr_larger_leaf_hist_data);

    // wait for GPU to finish, only if GPU is actually used

    if (is_gpu_used) {
      if (config_->gpu_use_dp) {
        // use double precision
        WaitAndGetHistograms<HistogramBinEntry>(ptr_larger_leaf_hist_data);
      }
      else {
        // use single precision
        WaitAndGetHistograms<GPUHistogramBinEntry>(ptr_larger_leaf_hist_data);
      }
    }
  }

  // ibmGBT
  #ifdef TIMETAG
  hist_time_ += std::chrono::steady_clock::now() - start_time;
  #endif
  nvtxRangePop();
}

void CUDATreeLearner::FindBestSplits() {
  nvtxRangePush(__PRETTY_FUNCTION__);
  SerialTreeLearner::FindBestSplits();

#if GPU_DEBUG >= 3
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!is_feature_used_[feature_index]) continue;
    if (parent_leaf_histogram_array_ != nullptr
        && !parent_leaf_histogram_array_[feature_index].is_splittable()) {
      smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }
    size_t bin_size = train_data_->FeatureNumBin(feature_index) + 1; 
    printf("Feature %d smaller leaf:\n", feature_index);
    PrintHistograms(smaller_leaf_histogram_array_[feature_index].RawData() - 1, bin_size);
    if (larger_leaf_splits_ == nullptr || larger_leaf_splits_->LeafIndex() < 0) { continue; }
    printf("Feature %d larger leaf:\n", feature_index);
    PrintHistograms(larger_leaf_histogram_array_[feature_index].RawData() - 1, bin_size);
  }
#endif
  nvtxRangePop();
}

void CUDATreeLearner::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  nvtxRangePush(__PRETTY_FUNCTION__);
  const SplitInfo& best_split_info = best_split_per_leaf_[best_Leaf];

#if GPU_DEBUG >= 2
  printf("Splitting leaf %d with feature %d thresh %d gain %f stat %f %f %f %f\n", best_Leaf, best_split_info.feature, best_split_info.threshold, best_split_info.gain, best_split_info.left_sum_gradient, best_split_info.right_sum_gradient, best_split_info.left_sum_hessian, best_split_info.right_sum_hessian);
#endif

  SerialTreeLearner::Split(tree, best_Leaf, left_leaf, right_leaf);

  if (Network::num_machines() == 1) {
    // do some sanity check for the GPU algorithm
    if (best_split_info.left_count < best_split_info.right_count) {
      if ((best_split_info.left_count != smaller_leaf_splits_->num_data_in_leaf()) ||
          (best_split_info.right_count!= larger_leaf_splits_->num_data_in_leaf())) {
        Log::Fatal("1 Bug in GPU histogram! split %d: %d, smaller_leaf: %d, larger_leaf: %d\n", best_split_info.left_count, best_split_info.right_count, smaller_leaf_splits_->num_data_in_leaf(), larger_leaf_splits_->num_data_in_leaf());
      }
    } else {
      double smaller_min = smaller_leaf_splits_->min_constraint();
      double smaller_max = smaller_leaf_splits_->max_constraint();
      double larger_min = larger_leaf_splits_->min_constraint();
      double larger_max = larger_leaf_splits_->max_constraint();
      smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(), best_split_info.right_sum_gradient, best_split_info.right_sum_hessian);
      larger_leaf_splits_->Init(*left_leaf, data_partition_.get(), best_split_info.left_sum_gradient, best_split_info.left_sum_hessian);
      smaller_leaf_splits_->SetValueConstraint(smaller_min, smaller_max);
      larger_leaf_splits_->SetValueConstraint(larger_min, larger_max);
      if ((best_split_info.left_count != larger_leaf_splits_->num_data_in_leaf()) ||
          (best_split_info.right_count!= smaller_leaf_splits_->num_data_in_leaf())) {
        Log::Fatal("2 Bug in GPU histogram! split %d: %d, smaller_leaf: %d, larger_leaf: %d\n", best_split_info.left_count, best_split_info.right_count, smaller_leaf_splits_->num_data_in_leaf(), larger_leaf_splits_->num_data_in_leaf());
      }
    }
  }

  nvtxRangePop();
}

}   // namespace LightGBM
#undef cudaMemcpy_DEBUG
#endif // USE_CUDA
