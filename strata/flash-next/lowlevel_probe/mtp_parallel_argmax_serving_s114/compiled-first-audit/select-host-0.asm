
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/mtp-parallel-argmax-main-s114-audit/verify_kernels.dp.cpp.o:     file format elf64-x86-64


Disassembly of section .text:

00000000000012a0 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_>:
    12a0:	55                   	push   %rbp
    12a1:	41 57                	push   %r15
    12a3:	41 56                	push   %r14
    12a5:	41 55                	push   %r13
    12a7:	41 54                	push   %r12
    12a9:	53                   	push   %rbx
    12aa:	48 81 ec e8 01 00 00 	sub    $0x1e8,%rsp
    12b1:	4d 89 c7             	mov    %r8,%r15
    12b4:	49 89 cc             	mov    %rcx,%r12
    12b7:	49 89 d5             	mov    %rdx,%r13
    12ba:	48 89 f5             	mov    %rsi,%rbp
    12bd:	49 89 fe             	mov    %rdi,%r14
    12c0:	48 8b 9c 24 30 02 00 00 	mov    0x230(%rsp),%rbx
    12c8:	48 85 db             	test   %rbx,%rbx
    12cb:	75 78                	jne    1345 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0xa5>
    12cd:	4c 89 4c 24 30       	mov    %r9,0x30(%rsp)
    12d2:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 12d9 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x39>	12d5: R_X86_64_PC32	_ZGVZN4dpct7dev_mgr8instanceEvE3d_m-0x4
    12d9:	84 c0                	test   %al,%al
    12db:	0f 84 5c 03 00 00    	je     163d <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x39d>
    12e1:	e8 00 00 00 00       	call   12e6 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x46>	12e2: R_X86_64_PLT32	_ZTHN4dpct7dev_mgr10_dev_stackE-0x4
    12e6:	64 48 8b 04 25 00 00 00 00 	mov    %fs:0x0,%rax	12eb: R_X86_64_TPOFF32	_ZN4dpct7dev_mgr10_dev_stackE+0x30
    12ef:	64 48 3b 04 25 00 00 00 00 	cmp    %fs:0x0,%rax	12f4: R_X86_64_TPOFF32	_ZN4dpct7dev_mgr10_dev_stackE+0x10
    12f8:	74 31                	je     132b <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x8b>
    12fa:	e8 00 00 00 00       	call   12ff <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x5f>	12fb: R_X86_64_PLT32	_ZTHN4dpct7dev_mgr10_dev_stackE-0x4
    12ff:	64 48 8b 04 25 00 00 00 00 	mov    %fs:0x0,%rax	1304: R_X86_64_TPOFF32	_ZN4dpct7dev_mgr10_dev_stackE+0x30
    1308:	64 48 3b 04 25 00 00 00 00 	cmp    %fs:0x0,%rax	130d: R_X86_64_TPOFF32	_ZN4dpct7dev_mgr10_dev_stackE+0x38
    1311:	75 12                	jne    1325 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x85>
    1313:	64 48 8b 0c 25 00 00 00 00 	mov    %fs:0x0,%rcx	1318: R_X86_64_TPOFF32	_ZN4dpct7dev_mgr10_dev_stackE+0x48
    131c:	b8 00 02 00 00       	mov    $0x200,%eax
    1321:	48 03 41 f8          	add    -0x8(%rcx),%rax
    1325:	48 83 c0 fc          	add    $0xfffffffffffffffc,%rax
    1329:	eb 05                	jmp    1330 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x90>
    132b:	b8 00 00 00 00       	mov    $0x0,%eax	132c: R_X86_64_32	_ZZN4dpct7dev_mgr8instanceEvE3d_m+0x40
    1330:	8b 30                	mov    (%rax),%esi
    1332:	bf 00 00 00 00       	mov    $0x0,%edi	1333: R_X86_64_32	_ZZN4dpct7dev_mgr8instanceEvE3d_m
    1337:	e8 00 00 00 00       	call   133c <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x9c>	1338: R_X86_64_PLT32	_ZNK4dpct7dev_mgr10get_deviceEj-0x4
    133c:	48 8b 58 08          	mov    0x8(%rax),%rbx
    1340:	4c 8b 4c 24 30       	mov    0x30(%rsp),%r9
    1345:	48 8b 84 24 40 02 00 00 	mov    0x240(%rsp),%rax
    134d:	48 8b 8c 24 38 02 00 00 	mov    0x238(%rsp),%rcx
    1355:	8b 94 24 28 02 00 00 	mov    0x228(%rsp),%edx
    135c:	48 8b b4 24 20 02 00 00 	mov    0x220(%rsp),%rsi
    1364:	4c 89 b4 24 c0 00 00 00 	mov    %r14,0xc0(%rsp)
    136c:	48 89 ac 24 c8 00 00 00 	mov    %rbp,0xc8(%rsp)
    1374:	4c 89 ac 24 d0 00 00 00 	mov    %r13,0xd0(%rsp)
    137c:	4c 89 a4 24 d8 00 00 00 	mov    %r12,0xd8(%rsp)
    1384:	4c 89 bc 24 e0 00 00 00 	mov    %r15,0xe0(%rsp)
    138c:	4c 89 8c 24 e8 00 00 00 	mov    %r9,0xe8(%rsp)
    1394:	48 89 b4 24 f0 00 00 00 	mov    %rsi,0xf0(%rsp)
    139c:	89 94 24 f8 00 00 00 	mov    %edx,0xf8(%rsp)
    13a3:	48 89 8c 24 00 01 00 00 	mov    %rcx,0x100(%rsp)
    13ab:	48 89 84 24 08 01 00 00 	mov    %rax,0x108(%rsp)
    13b3:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 13ba <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x11a>	13b6: R_X86_64_PC32	.rodata+0x4ffc
    13ba:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
    13c2:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 13c9 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x129>	13c5: R_X86_64_PC32	.rodata+0x4fec
    13c9:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
    13d1:	4c 8d 74 24 38       	lea    0x38(%rsp),%r14
    13d6:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
    13de:	4c 89 f7             	mov    %r14,%rdi
    13e1:	e8 00 00 00 00       	call   13e6 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x146>	13e2: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
    13e6:	4c 89 f7             	mov    %r14,%rdi
    13e9:	e8 00 00 00 00       	call   13ee <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x14e>	13ea: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
    13ee:	49 89 c6             	mov    %rax,%r14
    13f1:	48 c7 44 24 78 01 00 00 00 	movq   $0x1,0x78(%rsp)
    13fa:	48 c7 84 24 80 00 00 00 01 00 00 00 	movq   $0x1,0x80(%rsp)
    1406:	48 c7 84 24 88 00 00 00 00 01 00 00 	movq   $0x100,0x88(%rsp)
    1412:	48 c7 84 24 90 00 00 00 01 00 00 00 	movq   $0x1,0x90(%rsp)
    141e:	48 c7 84 24 98 00 00 00 01 00 00 00 	movq   $0x1,0x98(%rsp)
    142a:	48 c7 84 24 a0 00 00 00 00 01 00 00 	movq   $0x100,0xa0(%rsp)
    1436:	0f 57 c0             	xorps  %xmm0,%xmm0
    1439:	0f 11 84 24 a8 00 00 00 	movups %xmm0,0xa8(%rsp)
    1441:	48 c7 84 24 b8 00 00 00 00 00 00 00 	movq   $0x0,0xb8(%rsp)
    144d:	4c 8d 7c 24 78       	lea    0x78(%rsp),%r15
    1452:	4c 89 ff             	mov    %r15,%rdi
    1455:	e8 00 00 00 00       	call   145a <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x1ba>	1456: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
    145a:	48 8d 84 24 90 00 00 00 	lea    0x90(%rsp),%rax
    1462:	48 8d 8c 24 a8 00 00 00 	lea    0xa8(%rsp),%rcx
    146a:	4c 89 7c 24 58       	mov    %r15,0x58(%rsp)
    146f:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    1474:	48 89 4c 24 68       	mov    %rcx,0x68(%rsp)
    1479:	48 c7 44 24 70 03 00 00 00 	movq   $0x3,0x70(%rsp)
    1482:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
    1487:	4c 89 f6             	mov    %r14,%rsi
    148a:	e8 00 00 00 00       	call   148f <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x1ef>	148b: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
    148f:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 1496 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x1f6>	1492: R_X86_64_PC32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels24mtp_select_kernel_4611feEEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
    1496:	84 c0                	test   %al,%al
    1498:	0f 84 6d 01 00 00    	je     160b <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x36b>
    149e:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 14a5 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x205>	14a1: R_X86_64_PC32	_ZZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels24mtp_select_kernel_4611feEEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
    14a5:	c6 84 24 58 01 00 00 00 	movb   $0x0,0x158(%rsp)
    14ad:	c6 84 24 62 01 00 00 00 	movb   $0x0,0x162(%rsp)
    14b5:	c6 84 24 64 01 00 00 01 	movb   $0x1,0x164(%rsp)
    14bd:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
    14c5:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
    14cd:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
    14d5:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
    14dd:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
    14e5:	c6 84 24 94 01 00 00 00 	movb   $0x0,0x194(%rsp)
    14ed:	c6 84 24 a0 01 00 00 00 	movb   $0x0,0x1a0(%rsp)
    14f5:	c6 84 24 b8 01 00 00 00 	movb   $0x0,0x1b8(%rsp)
    14fd:	c6 84 24 d8 01 00 00 00 	movb   $0x0,0x1d8(%rsp)
    1505:	c6 84 24 e0 01 00 00 00 	movb   $0x0,0x1e0(%rsp)
    150d:	48 c7 44 24 48 00 00 00 00 	movq   $0x0,0x48(%rsp)	1512: R_X86_64_32S	.rodata+0x18a0
    1516:	48 8d 84 24 c0 00 00 00 	lea    0xc0(%rsp),%rax
    151e:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    1523:	0f 57 c0             	xorps  %xmm0,%xmm0
    1526:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
    152e:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
    1533:	e8 00 00 00 00       	call   1538 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x298>	1534: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
    1538:	31 c9                	xor    %ecx,%ecx
    153a:	80 7c 24 28 00       	cmpb   $0x0,0x28(%rsp)
    153f:	0f 94 c1             	sete   %cl
    1542:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
    154a:	0f 11 04 24          	movups %xmm0,(%rsp)
    154e:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
    1552:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    1557:	48 8d bc 24 40 01 00 00 	lea    0x140(%rsp),%rdi
    155f:	48 8d 54 24 58       	lea    0x58(%rsp),%rdx
    1564:	48 8d 4c 24 48       	lea    0x48(%rsp),%rcx
    1569:	4c 8d 8c 24 50 01 00 00 	lea    0x150(%rsp),%r9
    1571:	48 89 de             	mov    %rbx,%rsi
    1574:	4d 89 f0             	mov    %r14,%r8
    1577:	e8 00 00 00 00       	call   157c <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x2dc>	1578: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
    157c:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
    1581:	e8 00 00 00 00       	call   1586 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x2e6>	1582: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    1586:	48 8d 7c 24 38       	lea    0x38(%rsp),%rdi
    158b:	e8 00 00 00 00       	call   1590 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x2f0>	158c: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    1590:	48 8b 9c 24 48 01 00 00 	mov    0x148(%rsp),%rbx
    1598:	48 85 db             	test   %rbx,%rbx
    159b:	74 43                	je     15e0 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x340>
    159d:	48 8b 43 08          	mov    0x8(%rbx),%rax
    15a1:	48 b9 01 00 00 00 01 00 00 00 	movabs $0x100000001,%rcx
    15ab:	48 39 c8             	cmp    %rcx,%rax
    15ae:	75 1c                	jne    15cc <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x32c>
    15b0:	48 c7 43 08 00 00 00 00 	movq   $0x0,0x8(%rbx)
    15b8:	48 8b 03             	mov    (%rbx),%rax
    15bb:	48 89 df             	mov    %rbx,%rdi
    15be:	ff 50 10             	call   *0x10(%rax)
    15c1:	48 8b 03             	mov    (%rbx),%rax
    15c4:	48 89 df             	mov    %rbx,%rdi
    15c7:	ff 50 18             	call   *0x18(%rax)
    15ca:	eb 14                	jmp    15e0 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x340>
    15cc:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 15d3 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x333>	15ce: R_X86_64_PC32	__libc_single_threaded-0x5
    15d3:	74 1d                	je     15f2 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x352>
    15d5:	8d 48 ff             	lea    -0x1(%rax),%ecx
    15d8:	89 4b 08             	mov    %ecx,0x8(%rbx)
    15db:	83 f8 01             	cmp    $0x1,%eax
    15de:	74 21                	je     1601 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x361>
    15e0:	48 81 c4 e8 01 00 00 	add    $0x1e8,%rsp
    15e7:	5b                   	pop    %rbx
    15e8:	41 5c                	pop    %r12
    15ea:	41 5d                	pop    %r13
    15ec:	41 5e                	pop    %r14
    15ee:	41 5f                	pop    %r15
    15f0:	5d                   	pop    %rbp
    15f1:	c3                   	ret
    15f2:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    15f7:	f0 0f c1 43 08       	lock xadd %eax,0x8(%rbx)
    15fc:	83 f8 01             	cmp    $0x1,%eax
    15ff:	75 df                	jne    15e0 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x340>
    1601:	48 89 df             	mov    %rbx,%rdi
    1604:	e8 00 00 00 00       	call   1609 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x369>	1605: R_X86_64_PLT32	_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv-0x4
    1609:	eb d5                	jmp    15e0 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x340>
    160b:	bf 00 00 00 00       	mov    $0x0,%edi	160c: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels24mtp_select_kernel_4611feEEEEERNS1_16DeviceKernelInfoEvE4Info
    1610:	e8 00 00 00 00       	call   1615 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x375>	1611: R_X86_64_PLT32	__cxa_guard_acquire-0x4
    1615:	85 c0                	test   %eax,%eax
    1617:	0f 84 81 fe ff ff    	je     149e <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x1fe>
    161d:	bf 00 00 00 00       	mov    $0x0,%edi	161e: R_X86_64_32	_ZN4sycl3_V16detail27compile_time_kernel_info_v121CompileTimeKernelInfoI16dpct_kernel_nameIJN6strata7kernels24mtp_select_kernel_4611feEEEEE
    1622:	e8 00 00 00 00       	call   1627 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x387>	1623: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
    1627:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 162e <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x38e>	162a: R_X86_64_PC32	_ZZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels24mtp_select_kernel_4611feEEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
    162e:	bf 00 00 00 00       	mov    $0x0,%edi	162f: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels24mtp_select_kernel_4611feEEEEERNS1_16DeviceKernelInfoEvE4Info
    1633:	e8 00 00 00 00       	call   1638 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x398>	1634: R_X86_64_PLT32	__cxa_guard_release-0x4
    1638:	e9 61 fe ff ff       	jmp    149e <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x1fe>
    163d:	bf 00 00 00 00       	mov    $0x0,%edi	163e: R_X86_64_32	_ZGVZN4dpct7dev_mgr8instanceEvE3d_m
    1642:	e8 00 00 00 00       	call   1647 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x3a7>	1643: R_X86_64_PLT32	__cxa_guard_acquire-0x4
    1647:	85 c0                	test   %eax,%eax
    1649:	0f 84 92 fc ff ff    	je     12e1 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x41>
    164f:	bf 00 00 00 00       	mov    $0x0,%edi	1650: R_X86_64_32	_ZZN4dpct7dev_mgr8instanceEvE3d_m
    1654:	e8 00 00 00 00       	call   1659 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x3b9>	1655: R_X86_64_PLT32	_ZN4dpct7dev_mgrC2Ev-0x4
    1659:	bf 00 00 00 00       	mov    $0x0,%edi	165a: R_X86_64_32	_ZN4dpct7dev_mgrD2Ev
    165e:	be 00 00 00 00       	mov    $0x0,%esi	165f: R_X86_64_32	_ZZN4dpct7dev_mgr8instanceEvE3d_m
    1663:	ba 00 00 00 00       	mov    $0x0,%edx	1664: R_X86_64_32	__dso_handle
    1668:	e8 00 00 00 00       	call   166d <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x3cd>	1669: R_X86_64_PLT32	__cxa_atexit-0x4
    166d:	bf 00 00 00 00       	mov    $0x0,%edi	166e: R_X86_64_32	_ZGVZN4dpct7dev_mgr8instanceEvE3d_m
    1672:	e8 00 00 00 00       	call   1677 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x3d7>	1673: R_X86_64_PLT32	__cxa_guard_release-0x4
    1677:	e9 65 fc ff ff       	jmp    12e1 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x41>
    167c:	48 89 c3             	mov    %rax,%rbx
    167f:	bf 00 00 00 00       	mov    $0x0,%edi	1680: R_X86_64_32	_ZGVZN4dpct7dev_mgr8instanceEvE3d_m
    1684:	e8 00 00 00 00       	call   1689 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x3e9>	1685: R_X86_64_PLT32	__cxa_guard_abort-0x4
    1689:	48 89 df             	mov    %rbx,%rdi
    168c:	e8 00 00 00 00       	call   1691 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x3f1>	168d: R_X86_64_PLT32	_Unwind_Resume-0x4
    1691:	48 89 c3             	mov    %rax,%rbx
    1694:	bf 00 00 00 00       	mov    $0x0,%edi	1695: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels24mtp_select_kernel_4611feEEEEERNS1_16DeviceKernelInfoEvE4Info
    1699:	e8 00 00 00 00       	call   169e <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x3fe>	169a: R_X86_64_PLT32	__cxa_guard_abort-0x4
    169e:	eb 03                	jmp    16a3 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x403>
    16a0:	48 89 c3             	mov    %rax,%rbx
    16a3:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
    16a8:	e8 00 00 00 00       	call   16ad <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x40d>	16a9: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    16ad:	48 8d 7c 24 38       	lea    0x38(%rsp),%rdi
    16b2:	e8 00 00 00 00       	call   16b7 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x417>	16b3: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    16b7:	48 89 df             	mov    %rbx,%rdi
    16ba:	e8 00 00 00 00       	call   16bf <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x41f>	16bb: R_X86_64_PLT32	_Unwind_Resume-0x4
    16bf:	48 89 c3             	mov    %rax,%rbx
    16c2:	48 8d 7c 24 38       	lea    0x38(%rsp),%rdi
    16c7:	e8 00 00 00 00       	call   16cc <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x42c>	16c8: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    16cc:	48 89 df             	mov    %rbx,%rdi
    16cf:	e8 00 00 00 00       	call   16d4 <_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_+0x434>	16d0: R_X86_64_PLT32	_Unwind_Resume-0x4

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt5stackIjSt5dequeIjSaIjEEED2Ev:

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt3mapIPKvSt8functionIFvN4dpct4dim3ES4_PPvjPN4sycl3_V15queueEEESt4lessIS1_ESaISt4pairIKS1_SC_EEED2Ev:

Disassembly of section .text.__clang_call_terminate:

Disassembly of section .text._ZN6strata8spin_maxERKN4sycl3_V15queueE:

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

Disassembly of section .text._ZNK4dpct7dev_mgr10get_deviceEj:

Disassembly of section .text._ZTWN4dpct7dev_mgr10_dev_stackE:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev:

Disassembly of section .text._ZNSt13unordered_mapIN4sycl3_V16deviceEjSt4hashIS2_ESt8equal_toIS2_ESaISt4pairIKS2_jEEED2Ev:

Disassembly of section .text._ZZN6strata16intel_gpu_driverB5cxx11EvENKUlvE_clB5cxx11Ev:

Disassembly of section .text._ZNSt10filesystem7__cxx114pathD2Ev:

Disassembly of section .text._ZNSt10filesystem7__cxx1118directory_iteratorD2Ev:

Disassembly of section .text._ZNSt10filesystem7__cxx114pathC2IA7_cS1_EERKT_NS1_6formatE:

Disassembly of section .text._ZNSt10filesystem7__cxx114pathC2ERKS1_:

Disassembly of section .text._ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4findEPKcm:

Disassembly of section .text._ZNSt6thread24_M_thread_deps_never_runEv:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25fetch_blobs_kernel_14b6f5EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE:

Disassembly of section .text._ZN4sycl3_V19exceptionC1ESt10error_codePKc:

Disassembly of section .text._ZNSt12__shared_ptrIN4sycl3_V17contextELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25rebase_ptrs_kernel_436f27EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels35add_streams_broadcast_kernel_7de8f5EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels24ident_hits_kernel_287489EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels31copy_row_to_first_kernel_993785EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels24mtp_select_kernel_4611feEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25force_token_kernel_a9fa72EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25gather_rows_kernel_b3ae5aENS0_3vecIjLi4EEEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25gather_rows_kernel_f256acEjEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25gather_rows_kernel_c5b171EhEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels21map_ids_kernel_43a113EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26row_top_prob_kernel_6adbe2EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25argmax_rows_kernel_d5bcf6EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels32row_top_prob_split_kernel_9495b0EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels24window_ids_kernel_581536EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25dense_steps_kernel_fd7245EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels31gdn_conv_l2_multi_kernel_b3f737EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels29gdn_conv_commit_kernel_dc1d7fEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi1EES8_EEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi2EES7_ILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi3EES7_ILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi4EES7_ILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi5EES7_ILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi6EES7_ILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi8EES7_ILi0EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels30gdn_state_replay_kernel_e4d4c4E18dpct_kernel_scalarILi8EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels28gdn_step_split_kernel_83c676E18dpct_kernel_scalarILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_out_norm_kernel_a650ebE18dpct_kernel_scalarILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_out_norm_kernel_f6dfabE18dpct_kernel_scalarILi0EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels29gdn_step_commit_kernel_5d9da7EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels33gdn_step_norm_multi_kernel_42ae26E18dpct_kernel_scalarILi1EES8_EEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels33gdn_step_norm_multi_kernel_42ae26E18dpct_kernel_scalarILi0EES7_ILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels33gdn_step_norm_multi_kernel_42ae26E18dpct_kernel_scalarILi1EES7_ILi0EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels33gdn_step_norm_multi_kernel_42ae26E18dpct_kernel_scalarILi0EES8_EEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels27resident_plan_kernel_db6b7aEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels23resident_plan_ep_kernelEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels29wait_flag_ge_or_kernel_2b2de3EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels29copy_i32_unless_kernel_d9d761EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26copy_or_zero_kernel_5ef081EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26wait_flag_ge_kernel_d7debfEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels34embedding_gather_dev_kernel_600016EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels31broadcast_streams_kernel_aa4f6fEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26copy_indexed_kernel_b12e23EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail14HostKernelBaseD2Ev:
