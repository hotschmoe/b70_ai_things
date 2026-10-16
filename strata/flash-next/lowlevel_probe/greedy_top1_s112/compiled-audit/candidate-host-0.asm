
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/greedy-top1-s112-oracle-prep-v2/build/oracle.o:     file format elf64-x86-64


Disassembly of section .text:

00000000000051e0 <_ZN12_GLOBAL__N_15Route6recordEv>:
    51e0:	55                   	push   %rbp
    51e1:	41 57                	push   %r15
    51e3:	41 56                	push   %r14
    51e5:	41 54                	push   %r12
    51e7:	53                   	push   %rbx
    51e8:	48 81 ec 90 01 00 00 	sub    $0x190,%rsp
    51ef:	48 89 fb             	mov    %rdi,%rbx
    51f2:	80 7f 08 00          	cmpb   $0x0,0x8(%rdi)
    51f6:	0f 84 8e 02 00 00    	je     548a <_ZN12_GLOBAL__N_15Route6recordEv+0x2aa>
    51fc:	48 8b 7b 28          	mov    0x28(%rbx),%rdi
    5200:	44 8b 43 0c          	mov    0xc(%rbx),%r8d
    5204:	48 8b 4b 48          	mov    0x48(%rbx),%rcx
    5208:	48 8b b3 88 00 00 00 	mov    0x88(%rbx),%rsi
    520f:	41 8d 80 ff 0f 00 00 	lea    0xfff(%r8),%eax
    5216:	41 8d 90 fe 1f 00 00 	lea    0x1ffe(%r8),%edx
    521d:	85 c0                	test   %eax,%eax
    521f:	0f 49 d0             	cmovns %eax,%edx
    5222:	c1 fa 0c             	sar    $0xc,%edx
    5225:	83 fa 02             	cmp    $0x2,%edx
    5228:	bd 01 00 00 00       	mov    $0x1,%ebp
    522d:	0f 4d ea             	cmovge %edx,%ebp
    5230:	4c 8b 33             	mov    (%rbx),%r14
    5233:	81 fd 80 00 00 00    	cmp    $0x80,%ebp
    5239:	41 bc 80 00 00 00    	mov    $0x80,%r12d
    523f:	41 b9 80 00 00 00    	mov    $0x80,%r9d
    5245:	44 0f 4c cd          	cmovl  %ebp,%r9d
    5249:	43 8d 04 08          	lea    (%r8,%r9,1),%eax
    524d:	ff c8                	dec    %eax
    524f:	99                   	cltd
    5250:	41 f7 f9             	idiv   %r9d
    5253:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    5258:	44 89 44 24 18       	mov    %r8d,0x18(%rsp)
    525d:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
    5261:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    5266:	48 89 74 24 28       	mov    %rsi,0x28(%rsp)
    526b:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 5272 <_ZN12_GLOBAL__N_15Route6recordEv+0x92>	526e: R_X86_64_PC32	.rodata+0xbc
    5272:	48 89 84 24 60 01 00 00 	mov    %rax,0x160(%rsp)
    527a:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 5281 <_ZN12_GLOBAL__N_15Route6recordEv+0xa1>	527d: R_X86_64_PC32	.rodata+0xac
    5281:	0f 29 84 24 50 01 00 00 	movaps %xmm0,0x150(%rsp)
    5289:	4c 8d 7c 24 30       	lea    0x30(%rsp),%r15
    528e:	48 8d b4 24 50 01 00 00 	lea    0x150(%rsp),%rsi
    5296:	4c 89 ff             	mov    %r15,%rdi
    5299:	e8 00 00 00 00       	call   529e <_ZN12_GLOBAL__N_15Route6recordEv+0xbe>	529a: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
    529e:	4c 89 ff             	mov    %r15,%rdi
    52a1:	e8 00 00 00 00       	call   52a6 <_ZN12_GLOBAL__N_15Route6recordEv+0xc6>	52a2: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
    52a6:	81 fd 80 00 00 00    	cmp    $0x80,%ebp
    52ac:	44 0f 42 e5          	cmovb  %ebp,%r12d
    52b0:	41 c1 e4 08          	shl    $0x8,%r12d
    52b4:	48 c7 44 24 70 01 00 00 00 	movq   $0x1,0x70(%rsp)
    52bd:	48 c7 44 24 78 01 00 00 00 	movq   $0x1,0x78(%rsp)
    52c6:	4c 89 a4 24 80 00 00 00 	mov    %r12,0x80(%rsp)
    52ce:	48 8d 8c 24 88 00 00 00 	lea    0x88(%rsp),%rcx
    52d6:	48 c7 84 24 88 00 00 00 01 00 00 00 	movq   $0x1,0x88(%rsp)
    52e2:	48 c7 84 24 90 00 00 00 01 00 00 00 	movq   $0x1,0x90(%rsp)
    52ee:	48 c7 84 24 98 00 00 00 00 01 00 00 	movq   $0x100,0x98(%rsp)
    52fa:	48 8d 94 24 a0 00 00 00 	lea    0xa0(%rsp),%rdx
    5302:	0f 57 c0             	xorps  %xmm0,%xmm0
    5305:	0f 11 84 24 a0 00 00 00 	movups %xmm0,0xa0(%rsp)
    530d:	48 c7 84 24 b0 00 00 00 00 00 00 00 	movq   $0x0,0xb0(%rsp)
    5319:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
    531e:	48 89 74 24 50       	mov    %rsi,0x50(%rsp)
    5323:	48 89 4c 24 58       	mov    %rcx,0x58(%rsp)
    5328:	48 89 54 24 60       	mov    %rdx,0x60(%rsp)
    532d:	48 c7 44 24 68 03 00 00 00 	movq   $0x3,0x68(%rsp)
    5336:	48 89 e7             	mov    %rsp,%rdi
    5339:	48 89 c6             	mov    %rax,%rsi
    533c:	e8 00 00 00 00       	call   5341 <_ZN12_GLOBAL__N_15Route6recordEv+0x161>	533d: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
    5341:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 5348 <_ZN12_GLOBAL__N_15Route6recordEv+0x168>	5344: R_X86_64_PC32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
    5348:	84 c0                	test   %al,%al
    534a:	0f 84 8e 02 00 00    	je     55de <_ZN12_GLOBAL__N_15Route6recordEv+0x3fe>
    5350:	4c 8b 3d 00 00 00 00 	mov    0x0(%rip),%r15        # 5357 <_ZN12_GLOBAL__N_15Route6recordEv+0x177>	5353: R_X86_64_PC32	_ZZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
    5357:	c6 84 24 c0 00 00 00 00 	movb   $0x0,0xc0(%rsp)
    535f:	c6 84 24 ca 00 00 00 00 	movb   $0x0,0xca(%rsp)
    5367:	c6 84 24 cc 00 00 00 01 	movb   $0x1,0xcc(%rsp)
    536f:	c6 84 24 d4 00 00 00 00 	movb   $0x0,0xd4(%rsp)
    5377:	c6 84 24 dc 00 00 00 00 	movb   $0x0,0xdc(%rsp)
    537f:	c6 84 24 e4 00 00 00 00 	movb   $0x0,0xe4(%rsp)
    5387:	c6 84 24 ec 00 00 00 00 	movb   $0x0,0xec(%rsp)
    538f:	c6 84 24 f4 00 00 00 00 	movb   $0x0,0xf4(%rsp)
    5397:	c6 84 24 fc 00 00 00 00 	movb   $0x0,0xfc(%rsp)
    539f:	c6 84 24 08 01 00 00 00 	movb   $0x0,0x108(%rsp)
    53a7:	c6 84 24 20 01 00 00 00 	movb   $0x0,0x120(%rsp)
    53af:	c6 84 24 40 01 00 00 00 	movb   $0x0,0x140(%rsp)
    53b7:	c6 84 24 48 01 00 00 00 	movb   $0x0,0x148(%rsp)
    53bf:	48 c7 44 24 40 00 00 00 00 	movq   $0x0,0x40(%rsp)	53c4: R_X86_64_32S	_ZTVN4sycl3_V16detail13HostKernelRefIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_RSC_SB_Li3EEE+0x10
    53c8:	48 8d 44 24 10       	lea    0x10(%rsp),%rax
    53cd:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    53d2:	0f 57 c0             	xorps  %xmm0,%xmm0
    53d5:	0f 29 84 24 70 01 00 00 	movaps %xmm0,0x170(%rsp)
    53dd:	48 89 e7             	mov    %rsp,%rdi
    53e0:	e8 00 00 00 00       	call   53e5 <_ZN12_GLOBAL__N_15Route6recordEv+0x205>	53e1: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
    53e5:	31 c9                	xor    %ecx,%ecx
    53e7:	80 7c 24 08 00       	cmpb   $0x0,0x8(%rsp)
    53ec:	0f 94 c1             	sete   %cl
    53ef:	48 83 ec 20          	sub    $0x20,%rsp
    53f3:	0f 28 84 24 90 01 00 00 	movaps 0x190(%rsp),%xmm0
    53fb:	0f 11 04 24          	movups %xmm0,(%rsp)
    53ff:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
    5403:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    5408:	48 8d bc 24 a0 01 00 00 	lea    0x1a0(%rsp),%rdi
    5410:	48 8d 54 24 70       	lea    0x70(%rsp),%rdx
    5415:	48 8d 4c 24 60       	lea    0x60(%rsp),%rcx
    541a:	4c 8d 8c 24 d8 00 00 00 	lea    0xd8(%rsp),%r9
    5422:	4c 89 f6             	mov    %r14,%rsi
    5425:	4d 89 f8             	mov    %r15,%r8
    5428:	e8 00 00 00 00       	call   542d <_ZN12_GLOBAL__N_15Route6recordEv+0x24d>	5429: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
    542d:	48 83 c4 20          	add    $0x20,%rsp
    5431:	48 89 e7             	mov    %rsp,%rdi
    5434:	e8 00 00 00 00       	call   5439 <_ZN12_GLOBAL__N_15Route6recordEv+0x259>	5435: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    5439:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
    543e:	e8 00 00 00 00       	call   5443 <_ZN12_GLOBAL__N_15Route6recordEv+0x263>	543f: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    5443:	4c 8b b4 24 88 01 00 00 	mov    0x188(%rsp),%r14
    544b:	4d 85 f6             	test   %r14,%r14
    544e:	0f 84 e5 00 00 00    	je     5539 <_ZN12_GLOBAL__N_15Route6recordEv+0x359>
    5454:	49 8b 46 08          	mov    0x8(%r14),%rax
    5458:	48 b9 01 00 00 00 01 00 00 00 	movabs $0x100000001,%rcx
    5462:	48 39 c8             	cmp    %rcx,%rax
    5465:	0f 85 b1 00 00 00    	jne    551c <_ZN12_GLOBAL__N_15Route6recordEv+0x33c>
    546b:	49 c7 46 08 00 00 00 00 	movq   $0x0,0x8(%r14)
    5473:	49 8b 06             	mov    (%r14),%rax
    5476:	4c 89 f7             	mov    %r14,%rdi
    5479:	ff 50 10             	call   *0x10(%rax)
    547c:	49 8b 06             	mov    (%r14),%rax
    547f:	4c 89 f7             	mov    %r14,%rdi
    5482:	ff 50 18             	call   *0x18(%rax)
    5485:	e9 af 00 00 00       	jmp    5539 <_ZN12_GLOBAL__N_15Route6recordEv+0x359>
    548a:	c7 84 24 b8 00 00 00 14 00 00 00 	movl   $0x14,0xb8(%rsp)
    5495:	f3 0f 10 05 00 00 00 00 	movss  0x0(%rip),%xmm0        # 549d <_ZN12_GLOBAL__N_15Route6recordEv+0x2bd>	5499: R_X86_64_PC32	.LCPI50_0-0x4
    549d:	0f 13 84 24 bc 00 00 00 	movlps %xmm0,0xbc(%rsp)
    54a5:	c7 84 24 cc 00 00 00 00 00 00 00 	movl   $0x0,0xcc(%rsp)
    54b0:	f3 0f 10 05 00 00 00 00 	movss  0x0(%rip),%xmm0        # 54b8 <_ZN12_GLOBAL__N_15Route6recordEv+0x2d8>	54b4: R_X86_64_PC32	.LCPI50_1-0x4
    54b8:	0f 13 84 24 d0 00 00 00 	movlps %xmm0,0xd0(%rsp)
    54c0:	c7 84 24 d8 00 00 00 00 00 00 00 	movl   $0x0,0xd8(%rsp)
    54cb:	0f 57 c0             	xorps  %xmm0,%xmm0
    54ce:	0f 11 84 24 e0 00 00 00 	movups %xmm0,0xe0(%rsp)
    54d6:	c6 84 24 f0 00 00 00 01 	movb   $0x1,0xf0(%rsp)
    54de:	48 b8 00 00 00 00 01 00 00 00 	movabs $0x100000000,%rax
    54e8:	48 89 84 24 c4 00 00 00 	mov    %rax,0xc4(%rsp)
    54f0:	48 8b 7b 28          	mov    0x28(%rbx),%rdi
    54f4:	8b 53 0c             	mov    0xc(%rbx),%edx
    54f7:	4c 8d 8c 24 b8 00 00 00 	lea    0xb8(%rsp),%r9
    54ff:	be 01 00 00 00       	mov    $0x1,%esi
    5504:	31 c9                	xor    %ecx,%ecx
    5506:	45 31 c0             	xor    %r8d,%r8d
    5509:	ff 33                	push   (%rbx)
    550b:	ff b3 88 00 00 00    	push   0x88(%rbx)
    5511:	e8 00 00 00 00       	call   5516 <_ZN12_GLOBAL__N_15Route6recordEv+0x336>	5512: R_X86_64_PLT32	_ZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS0_13SamplerParamsEPiPv-0x4
    5516:	48 83 c4 10          	add    $0x10,%rsp
    551a:	eb 1d                	jmp    5539 <_ZN12_GLOBAL__N_15Route6recordEv+0x359>
    551c:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 5523 <_ZN12_GLOBAL__N_15Route6recordEv+0x343>	551e: R_X86_64_PC32	__libc_single_threaded-0x5
    5523:	0f 84 94 00 00 00    	je     55bd <_ZN12_GLOBAL__N_15Route6recordEv+0x3dd>
    5529:	8d 48 ff             	lea    -0x1(%rax),%ecx
    552c:	41 89 4e 08          	mov    %ecx,0x8(%r14)
    5530:	83 f8 01             	cmp    $0x1,%eax
    5533:	0f 84 98 00 00 00    	je     55d1 <_ZN12_GLOBAL__N_15Route6recordEv+0x3f1>
    5539:	48 8b 7b 28          	mov    0x28(%rbx),%rdi
    553d:	8b 53 0c             	mov    0xc(%rbx),%edx
    5540:	48 8b 8b 88 00 00 00 	mov    0x88(%rbx),%rcx
    5547:	4c 8b 83 a8 00 00 00 	mov    0xa8(%rbx),%r8
    554e:	4c 8b 4b 68          	mov    0x68(%rbx),%r9
    5552:	48 83 ec 08          	sub    $0x8,%rsp
    5556:	be 01 00 00 00       	mov    $0x1,%esi
    555b:	ff 33                	push   (%rbx)
    555d:	e8 00 00 00 00       	call   5562 <_ZN12_GLOBAL__N_15Route6recordEv+0x382>	555e: R_X86_64_PLT32	_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_-0x4
    5562:	48 83 c4 10          	add    $0x10,%rsp
    5566:	48 8b bb c8 00 00 00 	mov    0xc8(%rbx),%rdi
    556d:	48 8b 93 88 00 00 00 	mov    0x88(%rbx),%rdx
    5574:	48 8b 8b 08 01 00 00 	mov    0x108(%rbx),%rcx
    557b:	4c 8b 83 e8 00 00 00 	mov    0xe8(%rbx),%r8
    5582:	4c 8b 8b 28 01 00 00 	mov    0x128(%rbx),%r9
    5589:	48 83 ec 08          	sub    $0x8,%rsp
    558d:	be 00 28 00 00       	mov    $0x2800,%esi
    5592:	ff b3 68 01 00 00    	push   0x168(%rbx)
    5598:	ff b3 a8 00 00 00    	push   0xa8(%rbx)
    559e:	ff 33                	push   (%rbx)
    55a0:	6a 00                	push   $0x0
    55a2:	ff b3 48 01 00 00    	push   0x148(%rbx)
    55a8:	e8 00 00 00 00       	call   55ad <_ZN12_GLOBAL__N_15Route6recordEv+0x3cd>	55a9: R_X86_64_PLT32	_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_-0x4
    55ad:	48 81 c4 c0 01 00 00 	add    $0x1c0,%rsp
    55b4:	5b                   	pop    %rbx
    55b5:	41 5c                	pop    %r12
    55b7:	41 5e                	pop    %r14
    55b9:	41 5f                	pop    %r15
    55bb:	5d                   	pop    %rbp
    55bc:	c3                   	ret
    55bd:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    55c2:	f0 41 0f c1 46 08    	lock xadd %eax,0x8(%r14)
    55c8:	83 f8 01             	cmp    $0x1,%eax
    55cb:	0f 85 68 ff ff ff    	jne    5539 <_ZN12_GLOBAL__N_15Route6recordEv+0x359>
    55d1:	4c 89 f7             	mov    %r14,%rdi
    55d4:	e8 00 00 00 00       	call   55d9 <_ZN12_GLOBAL__N_15Route6recordEv+0x3f9>	55d5: R_X86_64_PLT32	_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv-0x4
    55d9:	e9 5b ff ff ff       	jmp    5539 <_ZN12_GLOBAL__N_15Route6recordEv+0x359>
    55de:	bf 00 00 00 00       	mov    $0x0,%edi	55df: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEERNS1_16DeviceKernelInfoEvE4Info
    55e3:	e8 00 00 00 00       	call   55e8 <_ZN12_GLOBAL__N_15Route6recordEv+0x408>	55e4: R_X86_64_PLT32	__cxa_guard_acquire-0x4
    55e8:	85 c0                	test   %eax,%eax
    55ea:	0f 84 60 fd ff ff    	je     5350 <_ZN12_GLOBAL__N_15Route6recordEv+0x170>
    55f0:	bf 00 00 00 00       	mov    $0x0,%edi	55f1: R_X86_64_32	_ZN4sycl3_V16detail27compile_time_kernel_info_v121CompileTimeKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEE
    55f5:	e8 00 00 00 00       	call   55fa <_ZN12_GLOBAL__N_15Route6recordEv+0x41a>	55f6: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
    55fa:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 5601 <_ZN12_GLOBAL__N_15Route6recordEv+0x421>	55fd: R_X86_64_PC32	_ZZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
    5601:	bf 00 00 00 00       	mov    $0x0,%edi	5602: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEERNS1_16DeviceKernelInfoEvE4Info
    5606:	e8 00 00 00 00       	call   560b <_ZN12_GLOBAL__N_15Route6recordEv+0x42b>	5607: R_X86_64_PLT32	__cxa_guard_release-0x4
    560b:	e9 40 fd ff ff       	jmp    5350 <_ZN12_GLOBAL__N_15Route6recordEv+0x170>
    5610:	48 89 c3             	mov    %rax,%rbx
    5613:	bf 00 00 00 00       	mov    $0x0,%edi	5614: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEERNS1_16DeviceKernelInfoEvE4Info
    5618:	e8 00 00 00 00       	call   561d <_ZN12_GLOBAL__N_15Route6recordEv+0x43d>	5619: R_X86_64_PLT32	__cxa_guard_abort-0x4
    561d:	eb 03                	jmp    5622 <_ZN12_GLOBAL__N_15Route6recordEv+0x442>
    561f:	48 89 c3             	mov    %rax,%rbx
    5622:	48 89 e7             	mov    %rsp,%rdi
    5625:	e8 00 00 00 00       	call   562a <_ZN12_GLOBAL__N_15Route6recordEv+0x44a>	5626: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    562a:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
    562f:	e8 00 00 00 00       	call   5634 <_ZN12_GLOBAL__N_15Route6recordEv+0x454>	5630: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    5634:	48 89 df             	mov    %rbx,%rdi
    5637:	e8 00 00 00 00       	call   563c <_ZN12_GLOBAL__N_15Route6recordEv+0x45c>	5638: R_X86_64_PLT32	_Unwind_Resume-0x4
    563c:	48 89 c3             	mov    %rax,%rbx
    563f:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
    5644:	e8 00 00 00 00       	call   5649 <_ZN12_GLOBAL__N_15Route6recordEv+0x469>	5645: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    5649:	48 89 df             	mov    %rbx,%rdi
    564c:	e8 00 00 00 00       	call   5651 <_ZN12_GLOBAL__N_15Route6recordEv+0x471>	564d: R_X86_64_PLT32	_Unwind_Resume-0x4

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt5stackIjSt5dequeIjSaIjEEED2Ev:

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt3mapIPKvSt8functionIFvN4dpct4dim3ES4_PPvjPN4sycl3_V15queueEEESt4lessIS1_ESaISt4pairIKS1_SC_EEED2Ev:

Disassembly of section .text._ZN4sycl3_V15queue14wait_and_throwERKNS0_6detail13code_locationE:

Disassembly of section .text.__clang_call_terminate:

Disassembly of section .text._ZNSt8_Rb_treeIPKvSt4pairIKS1_St8functionIFvN4dpct4dim3ES6_PPvjPN4sycl3_V15queueEEEESt10_Select1stISF_ESt4lessIS1_ESaISF_EE8_M_eraseEPSt13_Rb_tree_nodeISF_E:

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

Disassembly of section .text._ZN4sycl3_V13ext6oneapi12experimental6detail24executable_command_graphD2Ev:

Disassembly of section .text._ZN4sycl3_V110get_nativeILNS0_7backendE2ENS0_7contextEEENS0_14backend_traitsIXT_EE11return_typeIT0_EERKS7_:

Disassembly of section .text._ZNSt13unordered_setIPvSt4hashIS0_ESt8equal_toIS0_ESaIS0_EED2Ev:

Disassembly of section .text._ZN4sycl3_V19exceptionC1ESt10error_codePKc:

Disassembly of section .text._ZNSt12__shared_ptrIN4sycl3_V17contextELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZN4sycl3_V13ext6oneapi12experimental6detail24modifiable_command_graphD2Ev:

Disassembly of section .text._ZN4dpct12experimental6detail9graph_mgrD2Ev:

Disassembly of section .text._ZN4sycl3_V15eventD2Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_RSC_SB_Li3EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_RSC_SB_Li3EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_RSC_SB_Li3EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_SB_Li3EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail14HostKernelBaseD2Ev:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_SB_Li3EED0Ev:
