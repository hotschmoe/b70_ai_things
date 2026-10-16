
/mnt/vm_8tb/b70/build/strata-ab/src-ep/build-sycl-ep/CMakeFiles/strata_engine.dir/src/core/mtp.cpp.o:     file format elf64-x86-64


Disassembly of section .text:

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt5stackIjSt5dequeIjSaIjEEED2Ev:

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt3mapIPKvSt8functionIFvN4dpct4dim3ES4_PPvjPN4sycl3_V15queueEEESt4lessIS1_ESaISt4pairIKS1_SC_EEED2Ev:

Disassembly of section .text._ZN4sycl3_V15queue4waitERKNS0_6detail13code_locationE:

Disassembly of section .text.__clang_call_terminate:

Disassembly of section .text._ZN4sycl3_V13ext6oneapi12experimental6detail24executable_command_graphD2Ev:

Disassembly of section .text._ZN4dpct10device_ext13destroy_queueERPN4sycl3_V15queueE:

Disassembly of section .text._ZNSt6vectorIfSaIfEE6resizeEm:

Disassembly of section .text._ZN6strata21malloc_device_guardedEmRN4sycl3_V15queueEPKc:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEPKc:

Disassembly of section .text._ZN4sycl3_V15eventD2Ev:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSERKS4_:

Disassembly of section .text._ZNSt6vectorIN6strata4core10MtpDrafter6TensorESaIS3_EEaSERKS5_:

Disassembly of section .text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_:

Disassembly of section .text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEOS8_PKS5_:

Disassembly of section .text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_:

Disassembly of section .text._ZN6strata4core10MtpDrafter6TensorD2Ev:

Disassembly of section .text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEOS8_S9_:

Disassembly of section .text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_:

Disassembly of section .text._ZNSt7__cxx119to_stringEm:

Disassembly of section .text._ZN4dpct10device_ext21queues_wait_and_throwEv:

Disassembly of section .text._ZN6strata13big_fill_zeroERN4sycl3_V15queueEPvmm:

Disassembly of section .text._ZNSt6vectorIiSaIiEEC2EmRKS0_:

Disassembly of section .text._ZNK6strata4core10MtpDrafter10vocab_fileB5cxx11Ev:

Disassembly of section .text._ZN4sycl3_V16detail16PropertyListBaseD2Ev:

Disassembly of section .text._ZN6strata4core8OnDeviceC2Ei:

Disassembly of section .text._ZNSt6vectorIiSaIiEE6resizeEm:

Disassembly of section .text._ZN6strata4core8OnDeviceD2Ev:

Disassembly of section .text._ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE:

0000000000000000 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE>:
   0:	41 57                	push   %r15
   2:	41 56                	push   %r14
   4:	53                   	push   %rbx
   5:	48 81 ec c0 01 00 00 	sub    $0x1c0,%rsp
   c:	4c 89 c3             	mov    %r8,%rbx
   f:	49 89 d0             	mov    %rdx,%r8
  12:	8d 86 ff 0f 00 00    	lea    0xfff(%rsi),%eax
  18:	8d 96 fe 1f 00 00    	lea    0x1ffe(%rsi),%edx
  1e:	85 c0                	test   %eax,%eax
  20:	0f 49 d0             	cmovns %eax,%edx
  23:	c1 fa 0c             	sar    $0xc,%edx
  26:	83 fa 02             	cmp    $0x2,%edx
  29:	b8 01 00 00 00       	mov    $0x1,%eax
  2e:	0f 4d c2             	cmovge %edx,%eax
  31:	3d 80 00 00 00       	cmp    $0x80,%eax
  36:	41 bf 80 00 00 00    	mov    $0x80,%r15d
  3c:	44 0f 4c f8          	cmovl  %eax,%r15d
  40:	42 8d 04 3e          	lea    (%rsi,%r15,1),%eax
  44:	ff c8                	dec    %eax
  46:	99                   	cltd
  47:	41 f7 ff             	idiv   %r15d
  4a:	48 89 7c 24 38       	mov    %rdi,0x38(%rsp)
  4f:	89 74 24 40          	mov    %esi,0x40(%rsp)
  53:	89 44 24 44          	mov    %eax,0x44(%rsp)
  57:	4c 89 44 24 48       	mov    %r8,0x48(%rsp)
  5c:	48 89 4c 24 50       	mov    %rcx,0x50(%rsp)
  61:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 68 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x68>	64: R_X86_64_PC32	.rodata+0x7fc
  68:	48 89 84 24 f0 00 00 00 	mov    %rax,0xf0(%rsp)
  70:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 77 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x77>	73: R_X86_64_PC32	.rodata+0x7ec
  77:	0f 29 84 24 e0 00 00 00 	movaps %xmm0,0xe0(%rsp)
  7f:	4c 8d 74 24 58       	lea    0x58(%rsp),%r14
  84:	48 8d b4 24 e0 00 00 00 	lea    0xe0(%rsp),%rsi
  8c:	4c 89 f7             	mov    %r14,%rdi
  8f:	e8 00 00 00 00       	call   94 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x94>	90: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
  94:	4c 89 f7             	mov    %r14,%rdi
  97:	e8 00 00 00 00       	call   9c <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x9c>	98: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
  9c:	41 c1 e7 08          	shl    $0x8,%r15d
  a0:	48 c7 84 24 98 00 00 00 01 00 00 00 	movq   $0x1,0x98(%rsp)
  ac:	48 c7 84 24 a0 00 00 00 01 00 00 00 	movq   $0x1,0xa0(%rsp)
  b8:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
  c0:	48 8d 8c 24 b0 00 00 00 	lea    0xb0(%rsp),%rcx
  c8:	48 c7 84 24 b0 00 00 00 01 00 00 00 	movq   $0x1,0xb0(%rsp)
  d4:	48 c7 84 24 b8 00 00 00 01 00 00 00 	movq   $0x1,0xb8(%rsp)
  e0:	48 c7 84 24 c0 00 00 00 00 01 00 00 	movq   $0x100,0xc0(%rsp)
  ec:	48 8d 94 24 c8 00 00 00 	lea    0xc8(%rsp),%rdx
  f4:	0f 57 c0             	xorps  %xmm0,%xmm0
  f7:	0f 11 84 24 c8 00 00 00 	movups %xmm0,0xc8(%rsp)
  ff:	48 c7 84 24 d8 00 00 00 00 00 00 00 	movq   $0x0,0xd8(%rsp)
 10b:	48 8d b4 24 98 00 00 00 	lea    0x98(%rsp),%rsi
 113:	48 89 74 24 78       	mov    %rsi,0x78(%rsp)
 118:	48 89 8c 24 80 00 00 00 	mov    %rcx,0x80(%rsp)
 120:	48 89 94 24 88 00 00 00 	mov    %rdx,0x88(%rsp)
 128:	48 c7 84 24 90 00 00 00 03 00 00 00 	movq   $0x3,0x90(%rsp)
 134:	48 8d 7c 24 28       	lea    0x28(%rsp),%rdi
 139:	48 89 c6             	mov    %rax,%rsi
 13c:	e8 00 00 00 00       	call   141 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x141>	13d: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
 141:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 148 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x148>	144: R_X86_64_PC32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
 148:	84 c0                	test   %al,%al
 14a:	0f 84 65 01 00 00    	je     2b5 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x2b5>
 150:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 157 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x157>	153: R_X86_64_PC32	_ZZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
 157:	c6 84 24 30 01 00 00 00 	movb   $0x0,0x130(%rsp)
 15f:	c6 84 24 3a 01 00 00 00 	movb   $0x0,0x13a(%rsp)
 167:	c6 84 24 3c 01 00 00 01 	movb   $0x1,0x13c(%rsp)
 16f:	c6 84 24 44 01 00 00 00 	movb   $0x0,0x144(%rsp)
 177:	c6 84 24 4c 01 00 00 00 	movb   $0x0,0x14c(%rsp)
 17f:	c6 84 24 54 01 00 00 00 	movb   $0x0,0x154(%rsp)
 187:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
 18f:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
 197:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
 19f:	c6 84 24 78 01 00 00 00 	movb   $0x0,0x178(%rsp)
 1a7:	c6 84 24 90 01 00 00 00 	movb   $0x0,0x190(%rsp)
 1af:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
 1b7:	c6 84 24 b8 01 00 00 00 	movb   $0x0,0x1b8(%rsp)
 1bf:	48 c7 44 24 68 00 00 00 00 	movq   $0x0,0x68(%rsp)	1c4: R_X86_64_32S	_ZTVN4sycl3_V16detail13HostKernelRefIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_RSC_SB_Li3EEE+0x10
 1c8:	48 8d 44 24 38       	lea    0x38(%rsp),%rax
 1cd:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
 1d2:	0f 57 c0             	xorps  %xmm0,%xmm0
 1d5:	0f 29 84 24 00 01 00 00 	movaps %xmm0,0x100(%rsp)
 1dd:	48 8d 7c 24 28       	lea    0x28(%rsp),%rdi
 1e2:	e8 00 00 00 00       	call   1e7 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x1e7>	1e3: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
 1e7:	31 c9                	xor    %ecx,%ecx
 1e9:	80 7c 24 30 00       	cmpb   $0x0,0x30(%rsp)
 1ee:	0f 94 c1             	sete   %cl
 1f1:	0f 28 84 24 00 01 00 00 	movaps 0x100(%rsp),%xmm0
 1f9:	0f 11 04 24          	movups %xmm0,(%rsp)
 1fd:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
 201:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
 206:	48 8d bc 24 18 01 00 00 	lea    0x118(%rsp),%rdi
 20e:	48 8d 54 24 78       	lea    0x78(%rsp),%rdx
 213:	48 8d 4c 24 68       	lea    0x68(%rsp),%rcx
 218:	4c 8d 8c 24 28 01 00 00 	lea    0x128(%rsp),%r9
 220:	48 89 de             	mov    %rbx,%rsi
 223:	4d 89 f0             	mov    %r14,%r8
 226:	e8 00 00 00 00       	call   22b <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x22b>	227: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
 22b:	48 8d 7c 24 28       	lea    0x28(%rsp),%rdi
 230:	e8 00 00 00 00       	call   235 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x235>	231: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
 235:	48 8d 7c 24 58       	lea    0x58(%rsp),%rdi
 23a:	e8 00 00 00 00       	call   23f <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x23f>	23b: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
 23f:	48 8b 9c 24 20 01 00 00 	mov    0x120(%rsp),%rbx
 247:	48 85 db             	test   %rbx,%rbx
 24a:	74 43                	je     28f <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x28f>
 24c:	48 8b 43 08          	mov    0x8(%rbx),%rax
 250:	48 b9 01 00 00 00 01 00 00 00 	movabs $0x100000001,%rcx
 25a:	48 39 c8             	cmp    %rcx,%rax
 25d:	75 1c                	jne    27b <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x27b>
 25f:	48 c7 43 08 00 00 00 00 	movq   $0x0,0x8(%rbx)
 267:	48 8b 03             	mov    (%rbx),%rax
 26a:	48 89 df             	mov    %rbx,%rdi
 26d:	ff 50 10             	call   *0x10(%rax)
 270:	48 8b 03             	mov    (%rbx),%rax
 273:	48 89 df             	mov    %rbx,%rdi
 276:	ff 50 18             	call   *0x18(%rax)
 279:	eb 14                	jmp    28f <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x28f>
 27b:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 282 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x282>	27d: R_X86_64_PC32	__libc_single_threaded-0x5
 282:	74 18                	je     29c <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x29c>
 284:	8d 48 ff             	lea    -0x1(%rax),%ecx
 287:	89 4b 08             	mov    %ecx,0x8(%rbx)
 28a:	83 f8 01             	cmp    $0x1,%eax
 28d:	74 1c                	je     2ab <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x2ab>
 28f:	48 81 c4 c0 01 00 00 	add    $0x1c0,%rsp
 296:	5b                   	pop    %rbx
 297:	41 5e                	pop    %r14
 299:	41 5f                	pop    %r15
 29b:	c3                   	ret
 29c:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 2a1:	f0 0f c1 43 08       	lock xadd %eax,0x8(%rbx)
 2a6:	83 f8 01             	cmp    $0x1,%eax
 2a9:	75 e4                	jne    28f <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x28f>
 2ab:	48 89 df             	mov    %rbx,%rdi
 2ae:	e8 00 00 00 00       	call   2b3 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x2b3>	2af: R_X86_64_PLT32	_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv-0x4
 2b3:	eb da                	jmp    28f <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x28f>
 2b5:	bf 00 00 00 00       	mov    $0x0,%edi	2b6: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEERNS1_16DeviceKernelInfoEvE4Info
 2ba:	e8 00 00 00 00       	call   2bf <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x2bf>	2bb: R_X86_64_PLT32	__cxa_guard_acquire-0x4
 2bf:	85 c0                	test   %eax,%eax
 2c1:	0f 84 89 fe ff ff    	je     150 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x150>
 2c7:	bf 00 00 00 00       	mov    $0x0,%edi	2c8: R_X86_64_32	_ZN4sycl3_V16detail27compile_time_kernel_info_v121CompileTimeKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEE
 2cc:	e8 00 00 00 00       	call   2d1 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x2d1>	2cd: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
 2d1:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 2d8 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x2d8>	2d4: R_X86_64_PC32	_ZZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
 2d8:	bf 00 00 00 00       	mov    $0x0,%edi	2d9: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEERNS1_16DeviceKernelInfoEvE4Info
 2dd:	e8 00 00 00 00       	call   2e2 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x2e2>	2de: R_X86_64_PLT32	__cxa_guard_release-0x4
 2e2:	e9 69 fe ff ff       	jmp    150 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x150>
 2e7:	48 89 c3             	mov    %rax,%rbx
 2ea:	bf 00 00 00 00       	mov    $0x0,%edi	2eb: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEERNS1_16DeviceKernelInfoEvE4Info
 2ef:	e8 00 00 00 00       	call   2f4 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x2f4>	2f0: R_X86_64_PLT32	__cxa_guard_abort-0x4
 2f4:	eb 03                	jmp    2f9 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x2f9>
 2f6:	48 89 c3             	mov    %rax,%rbx
 2f9:	48 8d 7c 24 28       	lea    0x28(%rsp),%rdi
 2fe:	e8 00 00 00 00       	call   303 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x303>	2ff: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
 303:	48 8d 7c 24 58       	lea    0x58(%rsp),%rdi
 308:	e8 00 00 00 00       	call   30d <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x30d>	309: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
 30d:	48 89 df             	mov    %rbx,%rdi
 310:	e8 00 00 00 00       	call   315 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x315>	311: R_X86_64_PLT32	_Unwind_Resume-0x4
 315:	48 89 c3             	mov    %rax,%rbx
 318:	48 8d 7c 24 58       	lea    0x58(%rsp),%rdi
 31d:	e8 00 00 00 00       	call   322 <_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE+0x322>	31e: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
 322:	48 89 df             	mov    %rbx,%rdi
 325:	e8 00 00 00 00       	call   32a <GCC_except_table21+0xb6>	326: R_X86_64_PLT32	_Unwind_Resume-0x4

Disassembly of section .text._ZNSt7__cxx119to_stringEi:

Disassembly of section .text._ZN4sycl3_V113malloc_deviceIfEEPT_mRKNS0_5queueERKNS0_13property_listERKNS0_6detail13code_locationE:

Disassembly of section .text._ZN6strata4core13host_boundary4markEPKcii:

Disassembly of section .text._ZNSt8_Rb_treeIPKvSt4pairIKS1_St8functionIFvN4dpct4dim3ES6_PPvjPN4sycl3_V15queueEEEESt10_Select1stISF_ESt4lessIS1_ESaISF_EE8_M_eraseEPSt13_Rb_tree_nodeISF_E:

Disassembly of section .text._ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv:

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

Disassembly of section .text._ZNSt6vectorISt10shared_ptrIN4sycl3_V15queueEESaIS4_EED2Ev:

Disassembly of section .text._ZN4sycl3_V17contextD2Ev:

Disassembly of section .text._ZN4dpct10device_ext17create_queue_implIJN4sycl3_V18property5queue8in_orderEEEEPNS3_5queueEbDpT_:

Disassembly of section .text._ZNSt6vectorISt10shared_ptrIN4sycl3_V15queueEESaIS4_EE9push_backEOS4_:

Disassembly of section .text._ZNSt17_Function_handlerIFvN4sycl3_V114exception_listEEN4dpct17exception_handlerMUlS2_E_EE9_M_invokeERKSt9_Any_dataOS2_:

Disassembly of section .text._ZNSt17_Function_handlerIFvN4sycl3_V114exception_listEEN4dpct17exception_handlerMUlS2_E_EE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation:

Disassembly of section .text._ZN4sycl3_V114exception_listD2Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EED0Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_destroyEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info:

Disassembly of section .text._ZNK4dpct7dev_mgr8check_idEj:

Disassembly of section .text._ZTWN4dpct7dev_mgr10_dev_stackE:

Disassembly of section .text._ZNSt12__shared_ptrIN4sycl3_V15queueELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZN4sycl3_V110get_nativeILNS0_7backendE2ENS0_7contextEEENS0_14backend_traitsIXT_EE11return_typeIT0_EERKS7_:

Disassembly of section .text._ZNSt13unordered_setIPvSt4hashIS0_ESt8equal_toIS0_ESaIS0_EED2Ev:

Disassembly of section .text._ZN4sycl3_V19exceptionC1ESt10error_codePKc:

Disassembly of section .text._ZNSt12__shared_ptrIN4sycl3_V17contextELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZSt8_DestroyIPN6strata4core10MtpDrafter6TensorEEvT_S5_:

Disassembly of section .text._ZN6strata21device_alloc_registerEPKvm:

Disassembly of section .text._ZNSt3mapIPKvmSt4lessIS1_ESaISt4pairIKS1_mEEED2Ev:

Disassembly of section .text._ZNSt8_Rb_treeIPKvSt4pairIKS1_mESt10_Select1stIS4_ESt4lessIS1_ESaIS4_EE8_M_eraseEPSt13_Rb_tree_nodeIS4_E:

Disassembly of section .text._ZN4sycl3_V16detail15getRoundedRangeILi1EEESt5tupleIJNS0_5rangeIXT_EEEbEES5_RKNS0_6deviceE:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EESE_S5_Li1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EESE_S5_Li1EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EESE_S5_Li1EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EES5_Li1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EES5_Li1EED0Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_RSA_NS0_4itemILi1ELb1EEELi1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_RSA_NS0_4itemILi1ELb1EEELi1EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_RSA_NS0_4itemILi1ELb1EEELi1EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_NS0_4itemILi1ELb1EEELi1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_NS0_4itemILi1ELb1EEELi1EED0Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EESE_S5_Li1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EESE_S5_Li1EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EESE_S5_Li1EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EES5_Li1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EES5_Li1EED0Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_RSA_NS0_4itemILi1ELb1EEELi1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_RSA_NS0_4itemILi1ELb1EEELi1EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_RSA_NS0_4itemILi1ELb1EEELi1EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_NS0_4itemILi1ELb1EEELi1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_NS0_4itemILi1ELb1EEELi1EED0Ev:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm:

Disassembly of section .text._ZN6strata4core10MtpDrafter6TensorC2ERKS2_:

Disassembly of section .text._ZN4dpct10device_ext15get_memory_infoERmS1_:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EESD_S5_Li1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EESD_S5_Li1EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EESD_S5_Li1EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EES5_Li1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EES5_Li1EED0Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescIZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_RS9_NS0_4itemILi1ELb1EEELi1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_RS9_NS0_4itemILi1ELb1EEELi1EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefIZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_RS9_NS0_4itemILi1ELb1EEELi1EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_NS0_4itemILi1ELb1EEELi1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_NS0_4itemILi1ELb1EEELi1EED0Ev:

Disassembly of section .text._ZZN6strata4core19coupled_draft_stateEvENKUlvE_clEv:

Disassembly of section .text._ZNSt6vectorIN4sycl3_V15eventESaIS2_EED2Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescIN4dpct6detail21dpct_memcpy_3d_detailEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescINS1_19__pf_kernel_wrapperIN4dpct6detail21dpct_memcpy_3d_detailEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4dpct12sync_barrierEPN4sycl3_V15eventEPNS1_5queueE:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_RSC_SB_Li3EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_RSC_SB_Li3EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_RSC_SB_Li3EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_SB_Li3EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail14HostKernelBaseD2Ev:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_SB_Li3EED0Ev:

Disassembly of section .text._ZN4dpct12experimental6detail9graph_mgrD2Ev:

Disassembly of section .text._ZN4sycl3_V113malloc_deviceIiEEPT_mRKNS0_5queueERKNS0_13property_listERKNS0_6detail13code_locationE:

Disassembly of section .text._ZN6strata4core13host_boundary5StateD2Ev:
