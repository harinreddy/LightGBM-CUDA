#ifndef VECTOR_CH_H
#define VECTOR_CH_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <class T>
struct CHAllocator {
 typedef T value_type;
 CHAllocator() {}
 template <class U> CHAllocator(const CHAllocator<U>& other);
 T* allocate(std::size_t n)
 {
   T* ptr;
   if (n == 0) return NULL;
   #ifdef USE_CH
      cudaHostAlloc(&ptr, n*sizeof(T), cudaHostAllocPortable);
   #else
      ptr = (T*) malloc(n*sizeof(T));
   #endif
   return ptr;
 }
 void deallocate(T* p, std::size_t n)
 {
    if (p==NULL) return;
    #ifdef USE_CH
     cudaPointerAttributes attributes;
     cudaPointerGetAttributes (&attributes, p);
     if ((attributes.type == cudaMemoryTypeHost) && (attributes.devicePointer != NULL)){
        cudaFreeHost(p);
     }
    #else
        free(p);
    #endif
 }
};
template <class T, class U>
bool operator==(const CHAllocator<T>&, const CHAllocator<U>&);
template <class T, class U>
bool operator!=(const CHAllocator<T>&, const CHAllocator<U>&);
#endif
