
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/greedy-top1-s112-oracle-prep-v2-built-audit/sampler.dp.cpp.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000002b40 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_>:
    2b40:	55                   	push   %rbp
    2b41:	41 57                	push   %r15
    2b43:	41 56                	push   %r14
    2b45:	41 55                	push   %r13
    2b47:	41 54                	push   %r12
    2b49:	53                   	push   %rbx
    2b4a:	48 81 ec f8 01 00 00 	sub    $0x1f8,%rsp
    2b51:	49 89 f6             	mov    %rsi,%r14
    2b54:	48 89 fb             	mov    %rdi,%rbx
    2b57:	48 8b 07             	mov    (%rdi),%rax
    2b5a:	8b 30                	mov    (%rax),%esi
    2b5c:	48 c7 44 24 30 00 00 00 00 	movq   $0x0,0x30(%rsp)
    2b65:	48 c7 44 24 38 00 00 00 00 	movq   $0x0,0x38(%rsp)	2b6a: R_X86_64_32S	.rodata.str1.1+0x1b8
    2b6e:	48 b8 e7 05 00 00 36 00 00 00 	movabs $0x36000005e7,%rax
    2b78:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    2b7d:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    2b82:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    2b87:	0f 10 44 24 30       	movups 0x30(%rsp),%xmm0
    2b8c:	0f 11 04 24          	movups %xmm0,(%rsp)
    2b90:	48 8d bc 24 90 00 00 00 	lea    0x90(%rsp),%rdi
    2b98:	4c 89 f2             	mov    %r14,%rdx
    2b9b:	e8 00 00 00 00       	call   2ba0 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x60>	2b9c: R_X86_64_PLT32	_ZN4sycl3_V114local_accessorIhLi1EECI2NS0_19local_accessor_baseIhLi1ELNS0_6access4modeE1026ELNS4_11placeholderE0EEEILi1EvEENS0_5rangeILi1EEERNS0_7handlerENS0_6detail13code_locationE-0x4
    2ba0:	48 8b 43 08          	mov    0x8(%rbx),%rax
    2ba4:	8b 00                	mov    (%rax),%eax
    2ba6:	48 c1 e0 0a          	shl    $0xa,%rax
    2baa:	0f 57 c0             	xorps  %xmm0,%xmm0
    2bad:	0f 29 84 24 d0 00 00 00 	movaps %xmm0,0xd0(%rsp)
    2bb5:	48 c7 84 24 e0 00 00 00 00 00 00 00 	movq   $0x0,0xe0(%rsp)
    2bc1:	48 8b 4b 18          	mov    0x18(%rbx),%rcx
    2bc5:	48 8b 53 20          	mov    0x20(%rbx),%rdx
    2bc9:	48 8b 09             	mov    (%rcx),%rcx
    2bcc:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
    2bd1:	8b 2a                	mov    (%rdx),%ebp
    2bd3:	48 8b 4b 28          	mov    0x28(%rbx),%rcx
    2bd7:	4c 8b 21             	mov    (%rcx),%r12
    2bda:	48 8b 4b 30          	mov    0x30(%rbx),%rcx
    2bde:	44 8b 29             	mov    (%rcx),%r13d
    2be1:	48 8b 4b 38          	mov    0x38(%rbx),%rcx
    2be5:	0f 10 01             	movups (%rcx),%xmm0
    2be8:	0f 10 49 10          	movups 0x10(%rcx),%xmm1
    2bec:	0f 10 51 20          	movups 0x20(%rcx),%xmm2
    2bf0:	0f 10 59 29          	movups 0x29(%rcx),%xmm3
    2bf4:	0f 11 9c 24 bd 01 00 00 	movups %xmm3,0x1bd(%rsp)
    2bfc:	0f 11 94 24 b4 01 00 00 	movups %xmm2,0x1b4(%rsp)
    2c04:	0f 11 8c 24 a4 01 00 00 	movups %xmm1,0x1a4(%rsp)
    2c0c:	0f 11 84 24 94 01 00 00 	movups %xmm0,0x194(%rsp)
    2c14:	48 8b 4b 40          	mov    0x40(%rbx),%rcx
    2c18:	4c 8b 39             	mov    (%rcx),%r15
    2c1b:	0f 28 84 24 90 00 00 00 	movaps 0x90(%rsp),%xmm0
    2c23:	0f 29 84 24 d0 01 00 00 	movaps %xmm0,0x1d0(%rsp)
    2c2b:	48 8b 9c 24 98 00 00 00 	mov    0x98(%rsp),%rbx
    2c33:	48 85 db             	test   %rbx,%rbx
    2c36:	74 12                	je     2c4a <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x10a>
    2c38:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 2c3f <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0xff>	2c3a: R_X86_64_PC32	__libc_single_threaded-0x5
    2c3f:	74 05                	je     2c46 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x106>
    2c41:	ff 43 08             	incl   0x8(%rbx)
    2c44:	eb 04                	jmp    2c4a <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x10a>
    2c46:	f0 ff 43 08          	lock incl 0x8(%rbx)
    2c4a:	0f 28 84 24 a0 00 00 00 	movaps 0xa0(%rsp),%xmm0
    2c52:	0f 29 84 24 e0 01 00 00 	movaps %xmm0,0x1e0(%rsp)
    2c5a:	48 c7 44 24 48 01 00 00 00 	movq   $0x1,0x48(%rsp)
    2c63:	48 c7 44 24 50 01 00 00 00 	movq   $0x1,0x50(%rsp)
    2c6c:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    2c71:	48 c7 44 24 60 01 00 00 00 	movq   $0x1,0x60(%rsp)
    2c7a:	48 c7 44 24 68 01 00 00 00 	movq   $0x1,0x68(%rsp)
    2c83:	48 c7 44 24 70 00 04 00 00 	movq   $0x400,0x70(%rsp)
    2c8c:	0f 28 84 24 d0 00 00 00 	movaps 0xd0(%rsp),%xmm0
    2c94:	0f 11 44 24 78       	movups %xmm0,0x78(%rsp)
    2c99:	48 8b 84 24 e0 00 00 00 	mov    0xe0(%rsp),%rax
    2ca1:	48 89 84 24 88 00 00 00 	mov    %rax,0x88(%rsp)
    2ca9:	4c 89 f7             	mov    %r14,%rdi
    2cac:	e8 00 00 00 00       	call   2cb1 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x171>	2cad: R_X86_64_PLT32	_ZNK4sycl3_V17handler7getTypeEv-0x4
    2cb1:	85 c0                	test   %eax,%eax
    2cb3:	0f 85 21 03 00 00    	jne    2fda <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x49a>
    2cb9:	be 00 00 00 00       	mov    $0x0,%esi	2cba: R_X86_64_32	.rodata.str1.1+0x1d9
    2cbe:	ba 9f 00 00 00       	mov    $0x9f,%edx
    2cc3:	4c 89 f7             	mov    %r14,%rdi
    2cc6:	e8 00 00 00 00       	call   2ccb <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x18b>	2cc7: R_X86_64_PLT32	_ZN4sycl3_V17handler30verifyUsedKernelBundleInternalENS0_6detail11string_viewE-0x4
    2ccb:	48 8d 7c 24 48       	lea    0x48(%rsp),%rdi
    2cd0:	e8 00 00 00 00       	call   2cd5 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x195>	2cd1: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
    2cd5:	48 8b 84 24 88 00 00 00 	mov    0x88(%rsp),%rax
    2cdd:	48 89 84 24 30 01 00 00 	mov    %rax,0x130(%rsp)
    2ce5:	0f 10 44 24 48       	movups 0x48(%rsp),%xmm0
    2cea:	0f 10 4c 24 58       	movups 0x58(%rsp),%xmm1
    2cef:	0f 10 54 24 68       	movups 0x68(%rsp),%xmm2
    2cf4:	0f 10 5c 24 78       	movups 0x78(%rsp),%xmm3
    2cf9:	0f 29 9c 24 20 01 00 00 	movaps %xmm3,0x120(%rsp)
    2d01:	0f 29 94 24 10 01 00 00 	movaps %xmm2,0x110(%rsp)
    2d09:	0f 29 8c 24 00 01 00 00 	movaps %xmm1,0x100(%rsp)
    2d11:	0f 29 84 24 f0 00 00 00 	movaps %xmm0,0xf0(%rsp)
    2d19:	48 8d 84 24 20 01 00 00 	lea    0x120(%rsp),%rax
    2d21:	48 8d 8c 24 f0 00 00 00 	lea    0xf0(%rsp),%rcx
    2d29:	48 89 8c 24 b0 00 00 00 	mov    %rcx,0xb0(%rsp)
    2d31:	48 8d 8c 24 08 01 00 00 	lea    0x108(%rsp),%rcx
    2d39:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
    2d41:	48 89 84 24 c0 00 00 00 	mov    %rax,0xc0(%rsp)
    2d49:	48 c7 84 24 c8 00 00 00 03 00 00 00 	movq   $0x3,0xc8(%rsp)
    2d55:	0f 10 84 24 b0 00 00 00 	movups 0xb0(%rsp),%xmm0
    2d5d:	0f 10 8c 24 c0 00 00 00 	movups 0xc0(%rsp),%xmm1
    2d65:	0f 11 4c 24 10       	movups %xmm1,0x10(%rsp)
    2d6a:	0f 11 04 24          	movups %xmm0,(%rsp)
    2d6e:	4c 89 f7             	mov    %r14,%rdi
    2d71:	31 f6                	xor    %esi,%esi
    2d73:	e8 00 00 00 00       	call   2d78 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x238>	2d74: R_X86_64_PLT32	_ZN4sycl3_V17handler20setNDRangeDescriptorENS0_6detail16nd_range_view_v113nd_range_viewEb-0x4
    2d78:	bf 90 00 00 00       	mov    $0x90,%edi
    2d7d:	e8 00 00 00 00       	call   2d82 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x242>	2d7e: R_X86_64_PLT32	_Znwm-0x4
    2d82:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)	2d85: R_X86_64_32S	.rodata+0x4c0
    2d89:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
    2d8e:	48 89 48 08          	mov    %rcx,0x8(%rax)
    2d92:	89 68 10             	mov    %ebp,0x10(%rax)
    2d95:	4c 89 60 18          	mov    %r12,0x18(%rax)
    2d99:	44 89 68 20          	mov    %r13d,0x20(%rax)
    2d9d:	0f 10 84 24 90 01 00 00 	movups 0x190(%rsp),%xmm0
    2da5:	0f 10 8c 24 a0 01 00 00 	movups 0x1a0(%rsp),%xmm1
    2dad:	0f 10 94 24 b0 01 00 00 	movups 0x1b0(%rsp),%xmm2
    2db5:	0f 10 9c 24 bd 01 00 00 	movups 0x1bd(%rsp),%xmm3
    2dbd:	0f 11 40 24          	movups %xmm0,0x24(%rax)
    2dc1:	0f 11 48 34          	movups %xmm1,0x34(%rax)
    2dc5:	0f 11 50 44          	movups %xmm2,0x44(%rax)
    2dc9:	0f 11 58 51          	movups %xmm3,0x51(%rax)
    2dcd:	4c 89 78 68          	mov    %r15,0x68(%rax)
    2dd1:	0f 28 84 24 d0 01 00 00 	movaps 0x1d0(%rsp),%xmm0
    2dd9:	0f 11 40 70          	movups %xmm0,0x70(%rax)
    2ddd:	48 85 db             	test   %rbx,%rbx
    2de0:	74 12                	je     2df4 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x2b4>
    2de2:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 2de9 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x2a9>	2de4: R_X86_64_PC32	__libc_single_threaded-0x5
    2de9:	74 05                	je     2df0 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x2b0>
    2deb:	ff 43 08             	incl   0x8(%rbx)
    2dee:	eb 04                	jmp    2df4 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x2b4>
    2df0:	f0 ff 43 08          	lock incl 0x8(%rbx)
    2df4:	0f 28 84 24 e0 01 00 00 	movaps 0x1e0(%rsp),%xmm0
    2dfc:	0f 11 80 80 00 00 00 	movups %xmm0,0x80(%rax)
    2e03:	49 8b be 90 00 00 00 	mov    0x90(%r14),%rdi
    2e0a:	49 89 86 90 00 00 00 	mov    %rax,0x90(%r14)
    2e11:	48 85 ff             	test   %rdi,%rdi
    2e14:	74 0d                	je     2e23 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x2e3>
    2e16:	48 8b 07             	mov    (%rdi),%rax
    2e19:	ff 50 10             	call   *0x10(%rax)
    2e1c:	49 8b 86 90 00 00 00 	mov    0x90(%r14),%rax
    2e23:	48 8b 08             	mov    (%rax),%rcx
    2e26:	48 89 c7             	mov    %rax,%rdi
    2e29:	ff 11                	call   *(%rcx)
    2e2b:	49 c7 46 40 00 00 00 00 	movq   $0x0,0x40(%r14)	2e2f: R_X86_64_32S	.rodata.str1.1+0x1d9
    2e33:	49 c7 46 48 9f 00 00 00 	movq   $0x9f,0x48(%r14)
    2e3b:	4c 89 f7             	mov    %r14,%rdi
    2e3e:	48 89 c6             	mov    %rax,%rsi
    2e41:	e8 00 00 00 00       	call   2e46 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x306>	2e42: R_X86_64_PLT32	_ZN4sycl3_V17handler13setKernelFuncEPv-0x4
    2e46:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 2e4d <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x30d>	2e49: R_X86_64_PC32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels28sampler_greedy_kernel_5b2c3fEEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
    2e4d:	84 c0                	test   %al,%al
    2e4f:	0f 84 bd 01 00 00    	je     3012 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x4d2>
    2e55:	48 8b 35 00 00 00 00 	mov    0x0(%rip),%rsi        # 2e5c <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x31c>	2e58: R_X86_64_PC32	_ZZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels28sampler_greedy_kernel_5b2c3fEEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
    2e5c:	4c 89 f7             	mov    %r14,%rdi
    2e5f:	e8 00 00 00 00       	call   2e64 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x324>	2e60: R_X86_64_PLT32	_ZN4sycl3_V17handler22setDeviceKernelInfoPtrEPNS0_6detail16DeviceKernelInfoE-0x4
    2e64:	4c 89 f7             	mov    %r14,%rdi
    2e67:	be 01 00 00 00       	mov    $0x1,%esi
    2e6c:	e8 00 00 00 00       	call   2e71 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x331>	2e6d: R_X86_64_PLT32	_ZN4sycl3_V17handler7setTypeENS0_6detail6CGTypeE-0x4
    2e71:	c6 84 24 f8 00 00 00 00 	movb   $0x0,0xf8(%rsp)
    2e79:	c6 84 24 02 01 00 00 00 	movb   $0x0,0x102(%rsp)
    2e81:	c6 84 24 04 01 00 00 01 	movb   $0x1,0x104(%rsp)
    2e89:	c6 84 24 0c 01 00 00 00 	movb   $0x0,0x10c(%rsp)
    2e91:	c6 84 24 14 01 00 00 00 	movb   $0x0,0x114(%rsp)
    2e99:	c6 84 24 1c 01 00 00 00 	movb   $0x0,0x11c(%rsp)
    2ea1:	c6 84 24 24 01 00 00 00 	movb   $0x0,0x124(%rsp)
    2ea9:	c6 84 24 2c 01 00 00 00 	movb   $0x0,0x12c(%rsp)
    2eb1:	c6 84 24 34 01 00 00 00 	movb   $0x0,0x134(%rsp)
    2eb9:	c6 84 24 40 01 00 00 00 	movb   $0x0,0x140(%rsp)
    2ec1:	c6 84 24 58 01 00 00 00 	movb   $0x0,0x158(%rsp)
    2ec9:	c6 84 24 78 01 00 00 00 	movb   $0x0,0x178(%rsp)
    2ed1:	c6 84 24 80 01 00 00 00 	movb   $0x0,0x180(%rsp)
    2ed9:	48 8d b4 24 f0 00 00 00 	lea    0xf0(%rsp),%rsi
    2ee1:	4c 89 f7             	mov    %r14,%rdi
    2ee4:	e8 00 00 00 00       	call   2ee9 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x3a9>	2ee5: R_X86_64_PLT32	_ZN4sycl3_V17handler25setKernelLaunchPropertiesERKNS0_6detail27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENS5_5intel12experimental12cache_configENS7_17use_root_sync_keyENS7_23work_group_progress_keyENS7_22sub_group_progress_keyENS7_22work_item_progress_keyENS7_4cuda12cluster_sizeILi1EEENSH_ILi2EEENSH_ILi3EEEEEE-0x4
    2ee9:	49 be 01 00 00 00 01 00 00 00 	movabs $0x100000001,%r14
    2ef3:	48 85 db             	test   %rbx,%rbx
    2ef6:	74 49                	je     2f41 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x401>
    2ef8:	48 8b 43 08          	mov    0x8(%rbx),%rax
    2efc:	4c 39 f0             	cmp    %r14,%rax
    2eff:	75 2c                	jne    2f2d <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x3ed>
    2f01:	48 c7 43 08 00 00 00 00 	movq   $0x0,0x8(%rbx)
    2f09:	48 8b 03             	mov    (%rbx),%rax
    2f0c:	48 89 df             	mov    %rbx,%rdi
    2f0f:	ff 50 10             	call   *0x10(%rax)
    2f12:	48 8b 03             	mov    (%rbx),%rax
    2f15:	48 89 df             	mov    %rbx,%rdi
    2f18:	ff 50 18             	call   *0x18(%rax)
    2f1b:	48 8b 9c 24 98 00 00 00 	mov    0x98(%rsp),%rbx
    2f23:	48 85 db             	test   %rbx,%rbx
    2f26:	75 26                	jne    2f4e <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x40e>
    2f28:	e9 91 00 00 00       	jmp    2fbe <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x47e>
    2f2d:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 2f34 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x3f4>	2f2f: R_X86_64_PC32	__libc_single_threaded-0x5
    2f34:	74 53                	je     2f89 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x449>
    2f36:	8d 48 ff             	lea    -0x1(%rax),%ecx
    2f39:	89 4b 08             	mov    %ecx,0x8(%rbx)
    2f3c:	83 f8 01             	cmp    $0x1,%eax
    2f3f:	74 57                	je     2f98 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x458>
    2f41:	48 8b 9c 24 98 00 00 00 	mov    0x98(%rsp),%rbx
    2f49:	48 85 db             	test   %rbx,%rbx
    2f4c:	74 70                	je     2fbe <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x47e>
    2f4e:	48 8b 43 08          	mov    0x8(%rbx),%rax
    2f52:	4c 39 f0             	cmp    %r14,%rax
    2f55:	75 1c                	jne    2f73 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x433>
    2f57:	48 c7 43 08 00 00 00 00 	movq   $0x0,0x8(%rbx)
    2f5f:	48 8b 03             	mov    (%rbx),%rax
    2f62:	48 89 df             	mov    %rbx,%rdi
    2f65:	ff 50 10             	call   *0x10(%rax)
    2f68:	48 8b 03             	mov    (%rbx),%rax
    2f6b:	48 89 df             	mov    %rbx,%rdi
    2f6e:	ff 50 18             	call   *0x18(%rax)
    2f71:	eb 4b                	jmp    2fbe <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x47e>
    2f73:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 2f7a <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x43a>	2f75: R_X86_64_PC32	__libc_single_threaded-0x5
    2f7a:	74 33                	je     2faf <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x46f>
    2f7c:	8d 48 ff             	lea    -0x1(%rax),%ecx
    2f7f:	89 4b 08             	mov    %ecx,0x8(%rbx)
    2f82:	83 f8 01             	cmp    $0x1,%eax
    2f85:	75 37                	jne    2fbe <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x47e>
    2f87:	eb 47                	jmp    2fd0 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x490>
    2f89:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2f8e:	f0 0f c1 43 08       	lock xadd %eax,0x8(%rbx)
    2f93:	83 f8 01             	cmp    $0x1,%eax
    2f96:	75 a9                	jne    2f41 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x401>
    2f98:	48 89 df             	mov    %rbx,%rdi
    2f9b:	e8 00 00 00 00       	call   2fa0 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x460>	2f9c: R_X86_64_PLT32	_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv-0x4
    2fa0:	48 8b 9c 24 98 00 00 00 	mov    0x98(%rsp),%rbx
    2fa8:	48 85 db             	test   %rbx,%rbx
    2fab:	75 a1                	jne    2f4e <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x40e>
    2fad:	eb 0f                	jmp    2fbe <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x47e>
    2faf:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2fb4:	f0 0f c1 43 08       	lock xadd %eax,0x8(%rbx)
    2fb9:	83 f8 01             	cmp    $0x1,%eax
    2fbc:	74 12                	je     2fd0 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x490>
    2fbe:	48 81 c4 f8 01 00 00 	add    $0x1f8,%rsp
    2fc5:	5b                   	pop    %rbx
    2fc6:	41 5c                	pop    %r12
    2fc8:	41 5d                	pop    %r13
    2fca:	41 5e                	pop    %r14
    2fcc:	41 5f                	pop    %r15
    2fce:	5d                   	pop    %rbp
    2fcf:	c3                   	ret
    2fd0:	48 89 df             	mov    %rbx,%rdi
    2fd3:	e8 00 00 00 00       	call   2fd8 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x498>	2fd4: R_X86_64_PLT32	_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv-0x4
    2fd8:	eb e4                	jmp    2fbe <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x47e>
    2fda:	bf 40 00 00 00       	mov    $0x40,%edi
    2fdf:	e8 00 00 00 00       	call   2fe4 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x4a4>	2fe0: R_X86_64_PLT32	__cxa_allocate_exception-0x4
    2fe4:	49 89 c6             	mov    %rax,%r14
    2fe7:	bf 01 00 00 00       	mov    $0x1,%edi
    2fec:	e8 00 00 00 00       	call   2ff1 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x4b1>	2fed: R_X86_64_PLT32	_ZN4sycl3_V115make_error_codeENS0_4errcE-0x4
    2ff1:	b9 00 00 00 00       	mov    $0x0,%ecx	2ff2: R_X86_64_32	.rodata.str1.1+0x27a
    2ff6:	4c 89 f7             	mov    %r14,%rdi
    2ff9:	89 c6                	mov    %eax,%esi
    2ffb:	e8 00 00 00 00       	call   3000 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x4c0>	2ffc: R_X86_64_PLT32	_ZN4sycl3_V19exceptionC1ESt10error_codePKc-0x4
    3000:	be 00 00 00 00       	mov    $0x0,%esi	3001: R_X86_64_32	_ZTIN4sycl3_V19exceptionE
    3005:	ba 00 00 00 00       	mov    $0x0,%edx	3006: R_X86_64_32	_ZN4sycl3_V19exceptionD1Ev
    300a:	4c 89 f7             	mov    %r14,%rdi
    300d:	e8 00 00 00 00       	call   3012 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x4d2>	300e: R_X86_64_PLT32	__cxa_throw-0x4
    3012:	bf 00 00 00 00       	mov    $0x0,%edi	3013: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels28sampler_greedy_kernel_5b2c3fEEEEERNS1_16DeviceKernelInfoEvE4Info
    3017:	e8 00 00 00 00       	call   301c <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x4dc>	3018: R_X86_64_PLT32	__cxa_guard_acquire-0x4
    301c:	85 c0                	test   %eax,%eax
    301e:	0f 84 31 fe ff ff    	je     2e55 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x315>
    3024:	bf 00 00 00 00       	mov    $0x0,%edi	3025: R_X86_64_32	_ZN4sycl3_V16detail27compile_time_kernel_info_v121CompileTimeKernelInfoI16dpct_kernel_nameIJN6strata7kernels28sampler_greedy_kernel_5b2c3fEEEEE
    3029:	e8 00 00 00 00       	call   302e <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x4ee>	302a: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
    302e:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 3035 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x4f5>	3031: R_X86_64_PC32	_ZZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels28sampler_greedy_kernel_5b2c3fEEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
    3035:	bf 00 00 00 00       	mov    $0x0,%edi	3036: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels28sampler_greedy_kernel_5b2c3fEEEEERNS1_16DeviceKernelInfoEvE4Info
    303a:	e8 00 00 00 00       	call   303f <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x4ff>	303b: R_X86_64_PLT32	__cxa_guard_release-0x4
    303f:	e9 11 fe ff ff       	jmp    2e55 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x315>
    3044:	49 89 c7             	mov    %rax,%r15
    3047:	bf 00 00 00 00       	mov    $0x0,%edi	3048: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels28sampler_greedy_kernel_5b2c3fEEEEERNS1_16DeviceKernelInfoEvE4Info
    304c:	e8 00 00 00 00       	call   3051 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x511>	304d: R_X86_64_PLT32	__cxa_guard_abort-0x4
    3051:	eb 10                	jmp    3063 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x523>
    3053:	49 89 c7             	mov    %rax,%r15
    3056:	4c 89 f7             	mov    %r14,%rdi
    3059:	e8 00 00 00 00       	call   305e <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x51e>	305a: R_X86_64_PLT32	__cxa_free_exception-0x4
    305e:	eb 03                	jmp    3063 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x523>
    3060:	49 89 c7             	mov    %rax,%r15
    3063:	48 89 df             	mov    %rbx,%rdi
    3066:	e8 15 00 00 00       	call   3080 <_ZZZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS0_13SamplerParamsEPiPvENKUlRN4sycl3_V17handlerEE_clESD_ENUlNSB_7nd_itemILi3EEEE_D2Ev>
    306b:	48 8d bc 24 90 00 00 00 	lea    0x90(%rsp),%rdi
    3073:	e8 00 00 00 00       	call   3078 <_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerIZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS5_13SamplerParamsEPiPvEUlRNS0_7handlerEE_E4callEPKvSG_+0x538>	3074: R_X86_64_PLT32	_ZN4sycl3_V16detail21LocalAccessorBaseHostD2Ev-0x4
    3078:	4c 89 ff             	mov    %r15,%rdi
    307b:	e8 00 00 00 00       	call   3080 <_ZZZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS0_13SamplerParamsEPiPvENKUlRN4sycl3_V17handlerEE_clESD_ENUlNSB_7nd_itemILi3EEEE_D2Ev>	307c: R_X86_64_PLT32	_Unwind_Resume-0x4

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt5stackIjSt5dequeIjSaIjEEED2Ev:

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt3mapIPKvSt8functionIFvN4dpct4dim3ES4_PPvjPN4sycl3_V15queueEEESt4lessIS1_ESaISt4pairIKS1_SC_EEED2Ev:

Disassembly of section .text._ZN6strata4q_ofEPKv:

Disassembly of section .text._ZN4sycl3_V15eventD2Ev:

Disassembly of section .text.__clang_call_terminate:

Disassembly of section .text._ZNSt8_Rb_treeIPKvSt4pairIKS1_St8functionIFvN4dpct4dim3ES6_PPvjPN4sycl3_V15queueEEEESt10_Select1stISF_ESt4lessIS1_ESaISF_EE8_M_eraseEPSt13_Rb_tree_nodeISF_E:

Disassembly of section .text._ZN4dpct7dev_mgr14current_deviceEv:

Disassembly of section .text._ZN4dpct7dev_mgrC2Ev:

Disassembly of section .text._ZN4dpct7dev_mgrD2Ev:

Disassembly of section .text._ZNSt6vectorISt10shared_ptrIN4dpct10device_extEESaIS3_EE9push_backEOS3_:

Disassembly of section .text._ZSt11make_sharedIN4dpct10device_extEJRN4sycl3_V16deviceEEESt10shared_ptrIT_EDpOT0_:

Disassembly of section .text._ZNSt12__shared_ptrIN4dpct10device_extELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZNSt6vectorISt10shared_ptrIN4dpct10device_extEESaIS3_EED2Ev:

Disassembly of section .text._ZNSt17_Function_handlerIFiRKN4sycl3_V16deviceEEPS5_E9_M_invokeERKSt9_Any_dataS4_:

Disassembly of section .text._ZNSt17_Function_handlerIFiRKN4sycl3_V16deviceEEPS5_E10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation:

Disassembly of section .text._ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4dpct10device_extESaIvELN9__gnu_cxx12_Lock_policyE2EED0Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4dpct10device_extESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4dpct10device_extESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_destroyEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4dpct10device_extESaIvELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info:

Disassembly of section .text._ZN4sycl3_V16detail16PropertyListBaseD2Ev:

Disassembly of section .text._ZNSt6vectorISt10shared_ptrIN4sycl3_V15queueEESaIS4_EED2Ev:

Disassembly of section .text._ZN4sycl3_V17contextD2Ev:

Disassembly of section .text._ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv:

Disassembly of section .text._ZNSt6vectorISt10shared_ptrIN4sycl3_V15queueEESaIS4_EE9push_backEOS4_:

Disassembly of section .text._ZNSt12__shared_ptrIN4sycl3_V15queueELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZNSt17_Function_handlerIFvN4sycl3_V114exception_listEEN4dpct17exception_handlerMUlS2_E_EE9_M_invokeERKSt9_Any_dataOS2_:

Disassembly of section .text._ZNSt17_Function_handlerIFvN4sycl3_V114exception_listEEN4dpct17exception_handlerMUlS2_E_EE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation:

Disassembly of section .text._ZN4sycl3_V114exception_listD2Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EED0Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_destroyEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info:

Disassembly of section .text._ZTWN4dpct7dev_mgr10_dev_stackE:

Disassembly of section .text._ZNSt6vectorIPN4sycl3_V13vecIiLi2EEESaIS4_EED2Ev:

Disassembly of section .text._ZN4sycl3_V113malloc_deviceINS0_3vecIiLi2EEEEEPT_mRKNS0_5queueERKNS0_13property_listERKNS0_6detail13code_locationE:

Disassembly of section .text._ZN4sycl3_V114local_accessorIhLi1EECI2NS0_19local_accessor_baseIhLi1ELNS0_6access4modeE1026ELNS4_11placeholderE0EEEILi1EvEENS0_5rangeILi1EEERNS0_7handlerENS0_6detail13code_locationE:

Disassembly of section .text._ZN4sycl3_V16detail21LocalAccessorBaseHostD2Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels28sampler_greedy_kernel_5b2c3fEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE:

Disassembly of section .text._ZN4sycl3_V19exceptionC1ESt10error_codePKc:

Disassembly of section .text._ZNSt12__shared_ptrIN4sycl3_V17contextELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels21sampler_kernel_ebb5aaEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels32sampler_split_part_kernel_cc5673EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels33sampler_split_merge_kernel_b17bd4EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels31sampler_one_block_kernel_79fa88EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels27coupled_stage_kernel_113245EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels30coupled_penalize_kernel_6ca8a2EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels32sampler_split_part_kernel_cc5674EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels27coupled_merge_kernel_1fbeb1EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail14HostKernelBaseD2Ev:
