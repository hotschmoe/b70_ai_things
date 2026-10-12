#pragma once
#include "strata/core/original_mtp53_tensor_views.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <new>
#include <stdexcept>
#include <vector>
// Failed native drain retains ALL USM and graph handles until process retirement.
// _Exit86 intentionally prevents library destructors from freeing pending resources.
struct OwnedRuntime53 {
    sycl::queue& q;sycl::context context;std::vector<void*> buffers;std::vector<std::function<void()>> graph_deleters;bool retired=false;
    explicit OwnedRuntime53(sycl::queue& stream):q(stream),context(stream.get_context()){buffers.reserve(256);graph_deleters.reserve(32);}
    template<class T>T* take(size_t n){if(buffers.size()==256)throw std::runtime_error("MTP53 owner buffer bound");T* p=sycl::malloc_device<T>(n,q);if(!p)throw std::bad_alloc();buffers.push_back(p);return p;}
    template<class T>T* host(size_t n){if(buffers.size()==256)throw std::runtime_error("MTP53 owner buffer bound");T* p=sycl::malloc_host<T>(n,q);if(!p)throw std::bad_alloc();buffers.push_back(p);return p;}
    template<class T>T* keep_graph(T* p){if(graph_deleters.size()==32)throw std::runtime_error("MTP53 graph owner bound");graph_deleters.emplace_back([p]{delete p;});return p;}
    const void* upload(strata::core::original_mtp53::TensorViews& source,const char* name){auto v=source.at(name);auto* p=take<unsigned char>(v.bytes);q.memcpy(p,v.readonly_gguf_bytes,v.bytes);return p;}
    void retire()noexcept{
        if(retired)return;
        try{q.wait_and_throw();}catch(...){std::fprintf(stderr,"MTP53_FATAL_DRAIN retained_buffers=%zu retained_graphs=%zu\n",buffers.size(),graph_deleters.size());std::fflush(stderr);std::_Exit(86);}
        try{for(auto i=graph_deleters.rbegin();i!=graph_deleters.rend();++i)(*i)();graph_deleters.clear();for(void* p:buffers)sycl::free(p,context);buffers.clear();retired=true;}catch(...){std::fprintf(stderr,"MTP53_FATAL_RETIRE\n");std::fflush(stderr);std::_Exit(86);}
    }
    ~OwnedRuntime53(){retire();}
};
