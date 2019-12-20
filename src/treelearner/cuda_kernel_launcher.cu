#ifdef USE_CUDA

#include <nvToolsExt.h>
#include "cuda_kernel_launcher.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <LightGBM/utils/log.h>

using namespace LightGBM;

void cuda_histogram(
                data_size_t     leaf_num_data,
                data_size_t     num_data,
                bool            use_all_features,
                bool            is_constant_hessian,
                int             num_workgroups,
                cudaStream_t    stream,
                uint8_t*        arg0,
                uint8_t*        arg1,
                data_size_t     arg2,
                data_size_t*    arg3,
                data_size_t     arg4,
                score_t*        arg5,
                score_t*        arg6,
                score_t         arg6_const,
                char*           arg7,
                volatile int*   arg8,
                void*		arg9,
                size_t          exp_workgroups_per_feature) {

 if (leaf_num_data == num_data) {
   //Log::Debug("calling histogram_fulldata_kernel. exp_workgroups_per_feature=%d", exp_workgroups_per_feature);
   
   if (!is_constant_hessian) { 
     histogram256_fulldata<<<num_workgroups, 256, 0, stream>>>(
       arg0,
       arg1,
       arg2,
       reinterpret_cast<const uint*>(arg3),
       arg4,
       arg5,
       static_cast<float*>(arg6),
       arg7,
       arg8,
       static_cast<acc_type*>(arg9),
       exp_workgroups_per_feature);
   }
   else { 
     histogram256_fulldata<<<num_workgroups, 256, 0, stream>>>(
       arg0,
       arg1,
       arg2,
       reinterpret_cast<const uint*>(arg3),
       arg4,
       arg5,
       arg6_const, 
       arg7,
       arg8,
       static_cast<acc_type*>(arg9),
       exp_workgroups_per_feature);
   }

 }
 else {
   if (use_all_features) {
     // seems all features is always enabled, so this should be the same as fulldata
     //Log::Debug("calling histogram256 instead of allfeats. exp_workgroups_per_feature=%d", exp_workgroups_per_feature);
     
     if (!is_constant_hessian) { 
       histogram256<<<num_workgroups, 256, 0, stream>>>(
         arg0,
         arg1,
         arg2,
         reinterpret_cast<const uint*>(arg3),
         arg4,
         arg5,
         static_cast<float*>(arg6),
         arg7,
         arg8,
         static_cast<acc_type*>(arg9),
         exp_workgroups_per_feature
       );
     }
     else { 
       histogram256<<<num_workgroups, 256, 0, stream>>>(
         arg0,
         arg1,
         arg2,
         reinterpret_cast<const uint*>(arg3),
         arg4,
         arg5,
         arg6_const, 
         arg7,
         arg8,
         static_cast<acc_type*>(arg9),
         exp_workgroups_per_feature
       );
     } 
   }
   else {
     //Log::Debug("calling histogram256. exp_workgroups_per_feature=%d", exp_workgroups_per_feature);
     if (!is_constant_hessian) { 
       histogram256<<<num_workgroups, 256, 0, stream>>>(
         arg0,
         arg1,
         arg2,
         reinterpret_cast<const uint*>(arg3),
         arg4,
         arg5,
         static_cast<float*>(arg6),
         arg7,
         arg8,
         static_cast<acc_type*>(arg9),
         exp_workgroups_per_feature
       );
     }
     else { 
       histogram256<<<num_workgroups, 256, 0, stream>>>(
         arg0,
         arg1,
         arg2,
         reinterpret_cast<const uint*>(arg3),
         arg4,
         arg5,
         arg6_const, 
         arg7,
         arg8,
         static_cast<acc_type*>(arg9),
         exp_workgroups_per_feature
       );
     }
   }
 }
}

#endif // USE_CUDA
