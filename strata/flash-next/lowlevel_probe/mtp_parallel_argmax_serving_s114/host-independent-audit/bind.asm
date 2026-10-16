0000000000008730 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_>:
    8730:	55                   	push   %rbp
    8731:	41 57                	push   %r15
    8733:	41 56                	push   %r14
    8735:	41 55                	push   %r13
    8737:	41 54                	push   %r12
    8739:	53                   	push   %rbx
    873a:	48 83 ec 68          	sub    $0x68,%rsp
    873e:	4c 89 cb             	mov    %r9,%rbx
    8741:	4d 89 c4             	mov    %r8,%r12
    8744:	48 89 cd             	mov    %rcx,%rbp
    8747:	49 89 d7             	mov    %rdx,%r15
    874a:	49 89 f5             	mov    %rsi,%r13
    874d:	49 89 fe             	mov    %rdi,%r14
    8750:	8b b7 6c 01 00 00    	mov    0x16c(%rdi),%esi
    8756:	48 8d 7c 24 44       	lea    0x44(%rsp),%rdi
    875b:	e8 00 00 00 00       	call   8760 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x30>	875c: R_X86_64_PLT32	_ZN6strata4core8OnDeviceC2Ei-0x4
    8760:	4d 89 ae 50 01 00 00 	mov    %r13,0x150(%r14)
    8767:	4d 89 be 58 01 00 00 	mov    %r15,0x158(%r14)
    876e:	49 89 ae 60 01 00 00 	mov    %rbp,0x160(%r14)
    8775:	48 8d 6c 24 10       	lea    0x10(%rsp),%rbp
    877a:	48 89 2c 24          	mov    %rbp,(%rsp)
    877e:	48 b8 6f 75 74 70 75 74 2e 77 	movabs $0x772e74757074756f,%rax
    8788:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    878d:	48 b8 74 2e 77 65 69 67 68 74 	movabs $0x7468676965772e74,%rax
    8797:	48 89 44 24 15       	mov    %rax,0x15(%rsp)
    879c:	48 c7 44 24 08 0d 00 00 00 	movq   $0xd,0x8(%rsp)
    87a5:	c6 44 24 1d 00       	movb   $0x0,0x1d(%rsp)
    87aa:	48 89 e6             	mov    %rsp,%rsi
    87ad:	4c 89 ef             	mov    %r13,%rdi
    87b0:	e8 00 00 00 00       	call   87b5 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x85>	87b1: R_X86_64_PLT32	_ZNK6strata4core11WeightTable4findERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE-0x4
    87b5:	49 89 c5             	mov    %rax,%r13
    87b8:	48 8b 3c 24          	mov    (%rsp),%rdi
    87bc:	48 39 ef             	cmp    %rbp,%rdi
    87bf:	74 0d                	je     87ce <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x9e>
    87c1:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
    87c6:	48 ff c6             	inc    %rsi
    87c9:	e8 00 00 00 00       	call   87ce <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x9e>	87ca: R_X86_64_PLT32	_ZdlPvm-0x4
    87ce:	4d 85 ed             	test   %r13,%r13
    87d1:	0f 84 3f 01 00 00    	je     8916 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x1e6>
    87d7:	4d 8b 6d 18          	mov    0x18(%r13),%r13
    87db:	4d 89 ae c0 02 00 00 	mov    %r13,0x2c0(%r14)
    87e2:	41 b8 35 00 00 00    	mov    $0x35,%r8d
    87e8:	b9 00 00 00 00       	mov    $0x0,%ecx	87e9: R_X86_64_32	.rodata.str1.1+0x777
    87ed:	4d 85 ff             	test   %r15,%r15
    87f0:	0f 84 51 07 00 00    	je     8f47 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x817>
    87f6:	49 83 3f 00          	cmpq   $0x0,(%r15)
    87fa:	0f 84 47 07 00 00    	je     8f47 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x817>
    8800:	49 83 be c0 07 00 00 00 	cmpq   $0x0,0x7c0(%r14)
    8808:	0f 85 41 01 00 00    	jne    894f <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x21f>
    880e:	48 89 5c 24 60       	mov    %rbx,0x60(%rsp)
    8813:	49 63 9e 68 01 00 00 	movslq 0x168(%r14),%rbx
    881a:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 8821 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0xf1>	881d: R_X86_64_PC32	_ZGVZN4dpct7dev_mgr8instanceEvE3d_m-0x4
    8821:	84 c0                	test   %al,%al
    8823:	0f 84 91 05 00 00    	je     8dba <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x68a>
    8829:	bf 00 00 00 00       	mov    $0x0,%edi	882a: R_X86_64_32	_ZZN4dpct7dev_mgr8instanceEvE3d_m
    882e:	e8 00 00 00 00       	call   8833 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x103>	882f: R_X86_64_PLT32	_ZN4dpct7dev_mgr14current_deviceEv-0x4
    8833:	48 8b 70 08          	mov    0x8(%rax),%rsi
    8837:	4c 0f af eb          	imul   %rbx,%r13
    883b:	66 0f 57 c0          	xorpd  %xmm0,%xmm0
    883f:	66 0f 29 44 24 10    	movapd %xmm0,0x10(%rsp)
    8845:	66 0f 29 04 24       	movapd %xmm0,(%rsp)
    884a:	48 c7 44 24 20 00 00 00 00 	movq   $0x0,0x20(%rsp)
    8853:	48 c7 44 24 28 00 00 00 00 	movq   $0x0,0x28(%rsp)	8858: R_X86_64_32S	.rodata.str1.1+0x1ab8
    885c:	48 b8 ea 02 00 00 39 00 00 00 	movabs $0x39000002ea,%rax
    8866:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    886b:	48 89 e2             	mov    %rsp,%rdx
    886e:	48 8d 4c 24 20       	lea    0x20(%rsp),%rcx
    8873:	4c 89 ef             	mov    %r13,%rdi
    8876:	e8 00 00 00 00       	call   887b <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x14b>	8877: R_X86_64_PLT32	_ZN4sycl3_V113malloc_deviceIfEEPT_mRKNS0_5queueERKNS0_13property_listERKNS0_6detail13code_locationE-0x4
    887b:	49 89 86 c0 07 00 00 	mov    %rax,0x7c0(%r14)
    8882:	4c 8b 6c 24 08       	mov    0x8(%rsp),%r13
    8887:	48 8b 5c 24 10       	mov    0x10(%rsp),%rbx
    888c:	49 39 dd             	cmp    %rbx,%r13
    888f:	0f 84 a0 00 00 00    	je     8935 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x205>
    8895:	4c 89 64 24 58       	mov    %r12,0x58(%rsp)
    889a:	49 bc 01 00 00 00 01 00 00 00 	movabs $0x100000001,%r12
    88a4:	eb 2f                	jmp    88d5 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x1a5>
    88a6:	66 2e 0f 1f 84 00 00 00 00 00 	cs nopw 0x0(%rax,%rax,1)
    88b0:	48 c7 45 08 00 00 00 00 	movq   $0x0,0x8(%rbp)
    88b8:	48 8b 45 00          	mov    0x0(%rbp),%rax
    88bc:	48 89 ef             	mov    %rbp,%rdi
    88bf:	ff 50 10             	call   *0x10(%rax)
    88c2:	48 8b 45 00          	mov    0x0(%rbp),%rax
    88c6:	48 89 ef             	mov    %rbp,%rdi
    88c9:	ff 50 18             	call   *0x18(%rax)
    88cc:	49 83 c5 10          	add    $0x10,%r13
    88d0:	49 39 dd             	cmp    %rbx,%r13
    88d3:	74 51                	je     8926 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x1f6>
    88d5:	49 8b 6d 08          	mov    0x8(%r13),%rbp
    88d9:	48 85 ed             	test   %rbp,%rbp
    88dc:	74 ee                	je     88cc <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x19c>
    88de:	48 8b 45 08          	mov    0x8(%rbp),%rax
    88e2:	4c 39 e0             	cmp    %r12,%rax
    88e5:	74 c9                	je     88b0 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x180>
    88e7:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 88ee <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x1be>	88e9: R_X86_64_PC32	__libc_single_threaded-0x5
    88ee:	74 0d                	je     88fd <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x1cd>
    88f0:	8d 48 ff             	lea    -0x1(%rax),%ecx
    88f3:	89 4d 08             	mov    %ecx,0x8(%rbp)
    88f6:	83 f8 01             	cmp    $0x1,%eax
    88f9:	75 d1                	jne    88cc <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x19c>
    88fb:	eb 0f                	jmp    890c <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x1dc>
    88fd:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    8902:	f0 0f c1 45 08       	lock xadd %eax,0x8(%rbp)
    8907:	83 f8 01             	cmp    $0x1,%eax
    890a:	75 c0                	jne    88cc <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x19c>
    890c:	48 89 ef             	mov    %rbp,%rdi
    890f:	e8 00 00 00 00       	call   8914 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x1e4>	8910: R_X86_64_PLT32	_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv-0x4
    8914:	eb b6                	jmp    88cc <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x19c>
    8916:	41 b8 1d 00 00 00    	mov    $0x1d,%r8d
    891c:	b9 00 00 00 00       	mov    $0x0,%ecx	891d: R_X86_64_32	.rodata.str1.1+0x759
    8921:	e9 21 06 00 00       	jmp    8f47 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x817>
    8926:	4c 8b 6c 24 08       	mov    0x8(%rsp),%r13
    892b:	4c 8b 64 24 58       	mov    0x58(%rsp),%r12
    8930:	48 8d 6c 24 10       	lea    0x10(%rsp),%rbp
    8935:	4d 85 ed             	test   %r13,%r13
    8938:	48 8b 5c 24 60       	mov    0x60(%rsp),%rbx
    893d:	74 10                	je     894f <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x21f>
    893f:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
    8944:	4c 29 ee             	sub    %r13,%rsi
    8947:	4c 89 ef             	mov    %r13,%rdi
    894a:	e8 00 00 00 00       	call   894f <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x21f>	894b: R_X86_64_PLT32	_ZdlPvm-0x4
    894f:	48 85 db             	test   %rbx,%rbx
    8952:	74 78                	je     89cc <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x29c>
    8954:	41 b8 23 00 00 00    	mov    $0x23,%r8d
    895a:	b9 00 00 00 00       	mov    $0x0,%ecx	895b: R_X86_64_32	.rodata.str1.1+0x7ce
    895f:	4c 39 bb 58 01 00 00 	cmp    %r15,0x158(%rbx)
    8966:	0f 85 db 05 00 00    	jne    8f47 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x817>
    896c:	8b 83 6c 01 00 00    	mov    0x16c(%rbx),%eax
    8972:	41 3b 86 6c 01 00 00 	cmp    0x16c(%r14),%eax
    8979:	0f 85 c8 05 00 00    	jne    8f47 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x817>
    897f:	48 8b 83 c0 02 00 00 	mov    0x2c0(%rbx),%rax
    8986:	49 3b 86 c0 02 00 00 	cmp    0x2c0(%r14),%rax
    898d:	0f 85 b4 05 00 00    	jne    8f47 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x817>
    8993:	8b 83 98 01 00 00    	mov    0x198(%rbx),%eax
    8999:	41 89 86 98 01 00 00 	mov    %eax,0x198(%r14)
    89a0:	66 0f 10 83 f0 05 00 00 	movupd 0x5f0(%rbx),%xmm0
    89a8:	66 41 0f 11 86 f0 05 00 00 	movupd %xmm0,0x5f0(%r14)
    89b1:	48 8b 83 00 06 00 00 	mov    0x600(%rbx),%rax
    89b8:	49 89 86 00 06 00 00 	mov    %rax,0x600(%r14)
    89bf:	41 c6 86 01 04 00 00 00 	movb   $0x0,0x401(%r14)
    89c7:	e9 f8 00 00 00       	jmp    8ac4 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x394>
    89cc:	49 83 be f0 05 00 00 00 	cmpq   $0x0,0x5f0(%r14)
    89d4:	0f 85 ea 00 00 00    	jne    8ac4 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x394>
    89da:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 89e1 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x2b1>	89dd: R_X86_64_PC32	.bss+0x94
    89e1:	84 c0                	test   %al,%al
    89e3:	0f 84 56 04 00 00    	je     8e3f <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x70f>
    89e9:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 89f0 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x2c0>	89eb: R_X86_64_PC32	.bss+0x8b
    89f0:	0f 85 ce 00 00 00    	jne    8ac4 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x394>
    89f6:	4d 89 e5             	mov    %r12,%r13
    89f9:	66 0f 57 c0          	xorpd  %xmm0,%xmm0
    89fd:	66 0f 29 44 24 20    	movapd %xmm0,0x20(%rsp)
    8a03:	48 c7 44 24 30 00 00 00 00 	movq   $0x0,0x30(%rsp)
    8a0c:	48 89 e7             	mov    %rsp,%rdi
    8a0f:	4c 89 f6             	mov    %r14,%rsi
    8a12:	e8 00 00 00 00       	call   8a17 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x2e7>	8a13: R_X86_64_PLT32	_ZNK6strata4core10MtpDrafter10vocab_fileB5cxx11Ev-0x4
    8a17:	48 8b 3c 24          	mov    (%rsp),%rdi
    8a1b:	4c 8d 64 24 20       	lea    0x20(%rsp),%r12
    8a20:	4c 89 e6             	mov    %r12,%rsi
    8a23:	e8 28 c8 ff ff       	call   5250 <_ZN6strata4core12_GLOBAL__N_19read_fileERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIhSaIhEE>
    8a28:	84 c0                	test   %al,%al
    8a2a:	74 68                	je     8a94 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x364>
    8a2c:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
    8a31:	48 2b 5c 24 20       	sub    0x20(%rsp),%rbx
    8a36:	48 83 fb 04          	cmp    $0x4,%rbx
    8a3a:	72 58                	jb     8a94 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x364>
    8a3c:	48 8b 3c 24          	mov    (%rsp),%rdi
    8a40:	48 39 ef             	cmp    %rbp,%rdi
    8a43:	74 0d                	je     8a52 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x322>
    8a45:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
    8a4a:	48 ff c6             	inc    %rsi
    8a4d:	e8 00 00 00 00       	call   8a52 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x322>	8a4e: R_X86_64_PLT32	_ZdlPvm-0x4
    8a52:	f6 c3 03             	test   $0x3,%bl
    8a55:	75 53                	jne    8aaa <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x37a>
    8a57:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    8a5c:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
    8a61:	48 c1 e8 02          	shr    $0x2,%rax
    8a65:	49 89 86 00 06 00 00 	mov    %rax,0x600(%r14)
    8a6c:	49 63 4f 1c          	movslq 0x1c(%r15),%rcx
    8a70:	48 85 c9             	test   %rcx,%rcx
    8a73:	0f 8e 03 01 00 00    	jle    8b7c <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x44c>
    8a79:	49 8b 47 10          	mov    0x10(%r15),%rax
    8a7d:	48 89 c2             	mov    %rax,%rdx
    8a80:	48 c1 ea 20          	shr    $0x20,%rdx
    8a84:	0f 84 f6 00 00 00    	je     8b80 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x450>
    8a8a:	31 d2                	xor    %edx,%edx
    8a8c:	48 f7 f1             	div    %rcx
    8a8f:	e9 f0 00 00 00       	jmp    8b84 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x454>
    8a94:	48 8b 3c 24          	mov    (%rsp),%rdi
    8a98:	48 39 ef             	cmp    %rbp,%rdi
    8a9b:	74 0d                	je     8aaa <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x37a>
    8a9d:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
    8aa2:	48 ff c6             	inc    %rsi
    8aa5:	e8 00 00 00 00       	call   8aaa <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x37a>	8aa6: R_X86_64_PLT32	_ZdlPvm-0x4
    8aaa:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    8aaf:	48 85 ff             	test   %rdi,%rdi
    8ab2:	74 0d                	je     8ac1 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x391>
    8ab4:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
    8ab9:	48 29 fe             	sub    %rdi,%rsi
    8abc:	e8 00 00 00 00       	call   8ac1 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x391>	8abd: R_X86_64_PLT32	_ZdlPvm-0x4
    8ac1:	4d 89 ec             	mov    %r13,%r12
    8ac4:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 8acb <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x39b>	8ac7: R_X86_64_PC32	.bss+0xa4
    8acb:	84 c0                	test   %al,%al
    8acd:	0f 84 b8 02 00 00    	je     8d8b <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x65b>
    8ad3:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 8ada <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x3aa>	8ad5: R_X86_64_PC32	.bss+0x9b
    8ada:	74 63                	je     8b3f <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x40f>
    8adc:	49 83 be f0 05 00 00 00 	cmpq   $0x0,0x5f0(%r14)
    8ae4:	75 59                	jne    8b3f <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x40f>
    8ae6:	49 81 be c0 02 00 00 00 ca 03 00 	cmpq   $0x3ca00,0x2c0(%r14)
    8af1:	75 4c                	jne    8b3f <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x40f>
    8af3:	41 83 be 68 01 00 00 00 	cmpl   $0x0,0x168(%r14)
    8afb:	7e 30                	jle    8b2d <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x3fd>
    8afd:	49 83 be 40 06 00 00 00 	cmpq   $0x0,0x640(%r14)
    8b05:	74 26                	je     8b2d <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x3fd>
    8b07:	bf 01 00 00 00       	mov    $0x1,%edi
    8b0c:	e8 00 00 00 00       	call   8b11 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x3e1>	8b0d: R_X86_64_PLT32	_ZN6strata7kernels25argmax_rows_scratch_bytesEi-0x4
    8b11:	48 3d 10 04 00 00    	cmp    $0x410,%rax
    8b17:	75 14                	jne    8b2d <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x3fd>
    8b19:	41 8b be 68 01 00 00 	mov    0x168(%r14),%edi
    8b20:	e8 00 00 00 00       	call   8b25 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x3f5>	8b21: R_X86_64_PLT32	_ZN6strata7kernels25argmax_rows_scratch_bytesEi-0x4
    8b25:	48 3d 0f 04 00 00    	cmp    $0x40f,%rax
    8b2b:	77 12                	ja     8b3f <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x40f>
    8b2d:	be 00 00 00 00       	mov    $0x0,%esi	8b2e: R_X86_64_32	.rodata.str1.1+0x85b
    8b32:	4c 89 e7             	mov    %r12,%rdi
    8b35:	e8 00 00 00 00       	call   8b3a <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x40a>	8b36: R_X86_64_PLT32	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEPKc-0x4
    8b3a:	e9 c8 00 00 00       	jmp    8c07 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x4d7>
    8b3f:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 8b46 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x416>	8b42: R_X86_64_PC32	_ZGVZN6strata4core19coupled_draft_stateEvE2on-0x4
    8b46:	84 c0                	test   %al,%al
    8b48:	0f 84 ab 02 00 00    	je     8df9 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x6c9>
    8b4e:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 8b55 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x425>	8b50: R_X86_64_PC32	_ZZN6strata4core19coupled_draft_stateEvE2on-0x5
    8b55:	74 1d                	je     8b74 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x444>
    8b57:	49 83 be 08 01 00 00 00 	cmpq   $0x0,0x108(%r14)
    8b5f:	75 13                	jne    8b74 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x444>
    8b61:	4c 89 f7             	mov    %r14,%rdi
    8b64:	4c 89 e6             	mov    %r12,%rsi
    8b67:	e8 00 00 00 00       	call   8b6c <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x43c>	8b68: R_X86_64_PLT32	_ZN6strata4core10MtpDrafter13setup_coupledERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE-0x4
    8b6c:	84 c0                	test   %al,%al
    8b6e:	0f 84 93 00 00 00    	je     8c07 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x4d7>
    8b74:	41 b6 01             	mov    $0x1,%r14b
    8b77:	e9 dd 03 00 00       	jmp    8f59 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x829>
    8b7c:	31 c0                	xor    %eax,%eax
    8b7e:	eb 04                	jmp    8b84 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x454>
    8b80:	31 d2                	xor    %edx,%edx
    8b82:	f7 f1                	div    %ecx
    8b84:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    8b89:	4c 89 34 24          	mov    %r14,(%rsp)
    8b8d:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    8b92:	48 89 e7             	mov    %rsp,%rdi
    8b95:	e8 c6 04 00 00       	call   9060 <_ZZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_ENKUlvE0_clEv>
    8b9a:	85 c0                	test   %eax,%eax
    8b9c:	4c 89 eb             	mov    %r13,%rbx
    8b9f:	75 31                	jne    8bd2 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x4a2>
    8ba1:	4c 89 74 24 48       	mov    %r14,0x48(%rsp)
    8ba6:	48 8d 44 24 38       	lea    0x38(%rsp),%rax
    8bab:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    8bb0:	48 8d 7c 24 48       	lea    0x48(%rsp),%rdi
    8bb5:	e8 c6 05 00 00       	call   9180 <_ZZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_ENKUlvE1_clEv>
    8bba:	85 c0                	test   %eax,%eax
    8bbc:	75 14                	jne    8bd2 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x4a2>
    8bbe:	49 83 be f8 05 00 00 00 	cmpq   $0x0,0x5f8(%r14)
    8bc6:	74 0a                	je     8bd2 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x4a2>
    8bc8:	49 83 be f0 05 00 00 00 	cmpq   $0x0,0x5f0(%r14)
    8bd0:	75 3d                	jne    8c0f <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x4df>
    8bd2:	be 00 00 00 00       	mov    $0x0,%esi	8bd3: R_X86_64_32	.rodata.str1.1+0x7f2
    8bd7:	48 89 df             	mov    %rbx,%rdi
    8bda:	e8 00 00 00 00       	call   8bdf <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x4af>	8bdb: R_X86_64_PLT32	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEPKc-0x4
    8bdf:	49 8b be 00 06 00 00 	mov    0x600(%r14),%rdi
    8be6:	48 8b 74 24 38       	mov    0x38(%rsp),%rsi
    8beb:	e8 b0 06 00 00       	call   92a0 <_ZN6strata4core12_GLOBAL__N_115draft_head_hintEll>
    8bf0:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    8bf5:	48 85 ff             	test   %rdi,%rdi
    8bf8:	74 0d                	je     8c07 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x4d7>
    8bfa:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
    8bff:	48 29 fe             	sub    %rdi,%rsi
    8c02:	e8 00 00 00 00       	call   8c07 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x4d7>	8c03: R_X86_64_PLT32	_ZdlPvm-0x4
    8c07:	45 31 f6             	xor    %r14d,%r14d
    8c0a:	e9 4a 03 00 00       	jmp    8f59 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x829>
    8c0f:	e8 8c 8a ff ff       	call   16a0 <_ZN4dpctL18get_in_order_queueEv>
    8c14:	49 8b 96 f8 05 00 00 	mov    0x5f8(%r14),%rdx
    8c1b:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    8c20:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
    8c25:	49 29 c8             	sub    %rcx,%r8
    8c28:	48 c7 04 24 00 00 00 00 	movq   $0x0,(%rsp)
    8c30:	48 c7 44 24 08 00 00 00 00 	movq   $0x0,0x8(%rsp)	8c35: R_X86_64_32S	.rodata.str1.1+0x813
    8c39:	48 be 0b 03 00 00 28 00 00 00 	movabs $0x280000030b,%rsi
    8c43:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
    8c48:	48 8d 7c 24 48       	lea    0x48(%rsp),%rdi
    8c4d:	49 89 e1             	mov    %rsp,%r9
    8c50:	48 89 c6             	mov    %rax,%rsi
    8c53:	e8 00 00 00 00       	call   8c58 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x528>	8c54: R_X86_64_PLT32	_ZN4sycl3_V15queue6memcpyEPvPKvmRKNS0_6detail13code_locationE-0x4
    8c58:	48 8d 7c 24 48       	lea    0x48(%rsp),%rdi
    8c5d:	e8 00 00 00 00       	call   8c62 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x532>	8c5e: R_X86_64_PLT32	_ZN4sycl3_V15event4waitEv-0x4
    8c62:	48 8d 7c 24 48       	lea    0x48(%rsp),%rdi
    8c67:	e8 00 00 00 00       	call   8c6c <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x53c>	8c68: R_X86_64_PLT32	_ZN4sycl3_V15eventD2Ev-0x4
    8c6c:	49 8d be d8 01 00 00 	lea    0x1d8(%r14),%rdi
    8c73:	49 8b b6 00 06 00 00 	mov    0x600(%r14),%rsi
    8c7a:	e8 00 00 00 00       	call   8c7f <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x54f>	8c7b: R_X86_64_PLT32	_ZNSt6vectorIiSaIiEE6resizeEm-0x4
    8c7f:	49 8b be d8 01 00 00 	mov    0x1d8(%r14),%rdi
    8c86:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
    8c8b:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    8c90:	48 29 f2             	sub    %rsi,%rdx
    8c93:	e8 00 00 00 00       	call   8c98 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x568>	8c94: R_X86_64_PLT32	_intel_fast_memcpy-0x4
    8c98:	49 8b 3f             	mov    (%r15),%rdi
    8c9b:	48 8b 74 24 38       	mov    0x38(%rsp),%rsi
    8ca0:	4d 8b 86 f0 05 00 00 	mov    0x5f0(%r14),%r8
    8ca7:	49 8b 96 f8 05 00 00 	mov    0x5f8(%r14),%rdx
    8cae:	49 8b 8e 00 06 00 00 	mov    0x600(%r14),%rcx
    8cb5:	45 31 c9             	xor    %r9d,%r9d
    8cb8:	e8 00 00 00 00       	call   8cbd <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x58d>	8cb9: R_X86_64_PLT32	_ZN6strata7kernels11gather_rowsEPKhlPKilPhPv-0x4
    8cbd:	e8 4e 8a ff ff       	call   1710 <_ZN4dpctL18get_current_deviceEv>
    8cc2:	48 89 c7             	mov    %rax,%rdi
    8cc5:	e8 00 00 00 00       	call   8cca <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x59a>	8cc6: R_X86_64_PLT32	_ZN4dpct10device_ext21queues_wait_and_throwEv-0x4
    8cca:	49 8b 96 00 06 00 00 	mov    0x600(%r14),%rdx
    8cd1:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    8cd6:	48 0f af c2          	imul   %rdx,%rax
    8cda:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    8cdf:	48 03 44 24 28       	add    0x28(%rsp),%rax
    8ce4:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
    8ce9:	49 01 86 c8 02 00 00 	add    %rax,0x2c8(%r14)
    8cf0:	48 8b 3d 00 00 00 00 	mov    0x0(%rip),%rdi        # 8cf7 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x5c7>	8cf3: R_X86_64_PC32	stderr-0x4
    8cf7:	f2 0f 59 05 00 00 00 00 	mulsd  0x0(%rip),%xmm0        # 8cff <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x5cf>	8cfb: R_X86_64_PC32	.LCPI57_0-0x4
    8cff:	be 00 00 00 00       	mov    $0x0,%esi	8d00: R_X86_64_32	.rodata.str1.1+0x818
    8d04:	b0 01                	mov    $0x1,%al
    8d06:	e8 00 00 00 00       	call   8d0b <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x5db>	8d07: R_X86_64_PLT32	fprintf-0x4
    8d0b:	41 8b 47 20          	mov    0x20(%r15),%eax
    8d0f:	41 89 86 98 01 00 00 	mov    %eax,0x198(%r14)
    8d16:	41 80 be 76 01 00 00 00 	cmpb   $0x0,0x176(%r14)
    8d1e:	74 1a                	je     8d3a <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x60a>
    8d20:	4c 89 f7             	mov    %r14,%rdi
    8d23:	4c 89 ee             	mov    %r13,%rsi
    8d26:	e8 00 00 00 00       	call   8d2b <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x5fb>	8d27: R_X86_64_PLT32	_ZN6strata4core10MtpDrafter12make_q4_headERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE-0x4
    8d2b:	84 c0                	test   %al,%al
    8d2d:	0f 84 bd fe ff ff    	je     8bf0 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x4c0>
    8d33:	41 8b 86 98 01 00 00 	mov    0x198(%r14),%eax
    8d3a:	83 f8 0e             	cmp    $0xe,%eax
    8d3d:	0f 85 67 fd ff ff    	jne    8aaa <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x37a>
    8d43:	e8 00 00 00 00       	call   8d48 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x618>	8d44: R_X86_64_PLT32	_ZN6strata7kernels26native_q6_k_packed_enabledEv-0x4
    8d48:	84 c0                	test   %al,%al
    8d4a:	0f 84 5a fd ff ff    	je     8aaa <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x37a>
    8d50:	48 b8 05 4e e0 04 4e e0 04 4e 	movabs $0x4e04e04e04e04e05,%rax
    8d5a:	48 f7 6c 24 38       	imulq  0x38(%rsp)
    8d5f:	49 8b be f0 05 00 00 	mov    0x5f0(%r14),%rdi
    8d66:	48 89 d6             	mov    %rdx,%rsi
    8d69:	48 c1 ee 3f          	shr    $0x3f,%rsi
    8d6d:	c1 ea 06             	shr    $0x6,%edx
    8d70:	01 d6                	add    %edx,%esi
    8d72:	c1 e6 08             	shl    $0x8,%esi
    8d75:	41 8b 96 00 06 00 00 	mov    0x600(%r14),%edx
    8d7c:	b9 00 00 00 00       	mov    $0x0,%ecx	8d7d: R_X86_64_32	.rodata.str1.1+0x84c
    8d81:	e8 00 00 00 00       	call   8d86 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x656>	8d82: R_X86_64_PLT32	_ZN6strata7kernels16native_q6_k_packEPKviiPKc-0x4
    8d86:	e9 1f fd ff ff       	jmp    8aaa <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x37a>
    8d8b:	bf 00 00 00 00       	mov    $0x0,%edi	8d8c: R_X86_64_32	.bss+0xa8
    8d90:	e8 00 00 00 00       	call   8d95 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x665>	8d91: R_X86_64_PLT32	__cxa_guard_acquire-0x4
    8d95:	85 c0                	test   %eax,%eax
    8d97:	0f 84 36 fd ff ff    	je     8ad3 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x3a3>
    8d9d:	bf 00 00 00 00       	mov    $0x0,%edi	8d9e: R_X86_64_32	.rodata.str1.1+0x1dc3
    8da2:	e8 00 00 00 00       	call   8da7 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x677>	8da3: R_X86_64_PLT32	getenv-0x4
    8da7:	48 85 c0             	test   %rax,%rax
    8daa:	74 7c                	je     8e28 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x6f8>
    8dac:	80 38 31             	cmpb   $0x31,(%rax)
    8daf:	75 77                	jne    8e28 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x6f8>
    8db1:	80 78 01 00          	cmpb   $0x0,0x1(%rax)
    8db5:	0f 94 c0             	sete   %al
    8db8:	eb 70                	jmp    8e2a <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x6fa>
    8dba:	bf 00 00 00 00       	mov    $0x0,%edi	8dbb: R_X86_64_32	_ZGVZN4dpct7dev_mgr8instanceEvE3d_m
    8dbf:	e8 00 00 00 00       	call   8dc4 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x694>	8dc0: R_X86_64_PLT32	__cxa_guard_acquire-0x4
    8dc4:	85 c0                	test   %eax,%eax
    8dc6:	0f 84 5d fa ff ff    	je     8829 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0xf9>
    8dcc:	bf 00 00 00 00       	mov    $0x0,%edi	8dcd: R_X86_64_32	_ZZN4dpct7dev_mgr8instanceEvE3d_m
    8dd1:	e8 00 00 00 00       	call   8dd6 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x6a6>	8dd2: R_X86_64_PLT32	_ZN4dpct7dev_mgrC2Ev-0x4
    8dd6:	bf 00 00 00 00       	mov    $0x0,%edi	8dd7: R_X86_64_32	_ZN4dpct7dev_mgrD2Ev
    8ddb:	be 00 00 00 00       	mov    $0x0,%esi	8ddc: R_X86_64_32	_ZZN4dpct7dev_mgr8instanceEvE3d_m
    8de0:	ba 00 00 00 00       	mov    $0x0,%edx	8de1: R_X86_64_32	__dso_handle
    8de5:	e8 00 00 00 00       	call   8dea <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x6ba>	8de6: R_X86_64_PLT32	__cxa_atexit-0x4
    8dea:	bf 00 00 00 00       	mov    $0x0,%edi	8deb: R_X86_64_32	_ZGVZN4dpct7dev_mgr8instanceEvE3d_m
    8def:	e8 00 00 00 00       	call   8df4 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x6c4>	8df0: R_X86_64_PLT32	__cxa_guard_release-0x4
    8df4:	e9 30 fa ff ff       	jmp    8829 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0xf9>
    8df9:	bf 00 00 00 00       	mov    $0x0,%edi	8dfa: R_X86_64_32	_ZGVZN6strata4core19coupled_draft_stateEvE2on
    8dfe:	e8 00 00 00 00       	call   8e03 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x6d3>	8dff: R_X86_64_PLT32	__cxa_guard_acquire-0x4
    8e03:	85 c0                	test   %eax,%eax
    8e05:	0f 84 43 fd ff ff    	je     8b4e <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x41e>
    8e0b:	48 89 e7             	mov    %rsp,%rdi
    8e0e:	e8 00 00 00 00       	call   8e13 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x6e3>	8e0f: R_X86_64_PLT32	_ZZN6strata4core19coupled_draft_stateEvENKUlvE_clEv-0x4
    8e13:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 8e19 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x6e9>	8e15: R_X86_64_PC32	_ZZN6strata4core19coupled_draft_stateEvE2on-0x4
    8e19:	bf 00 00 00 00       	mov    $0x0,%edi	8e1a: R_X86_64_32	_ZGVZN6strata4core19coupled_draft_stateEvE2on
    8e1e:	e8 00 00 00 00       	call   8e23 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x6f3>	8e1f: R_X86_64_PLT32	__cxa_guard_release-0x4
    8e23:	e9 26 fd ff ff       	jmp    8b4e <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x41e>
    8e28:	31 c0                	xor    %eax,%eax
    8e2a:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 8e30 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x700>	8e2c: R_X86_64_PC32	.bss+0x9c
    8e30:	bf 00 00 00 00       	mov    $0x0,%edi	8e31: R_X86_64_32	.bss+0xa8
    8e35:	e8 00 00 00 00       	call   8e3a <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x70a>	8e36: R_X86_64_PLT32	__cxa_guard_release-0x4
    8e3a:	e9 94 fc ff ff       	jmp    8ad3 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x3a3>
    8e3f:	bf 00 00 00 00       	mov    $0x0,%edi	8e40: R_X86_64_32	.bss+0x98
    8e44:	e8 00 00 00 00       	call   8e49 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x719>	8e45: R_X86_64_PLT32	__cxa_guard_acquire-0x4
    8e49:	85 c0                	test   %eax,%eax
    8e4b:	0f 84 98 fb ff ff    	je     89e9 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x2b9>
    8e51:	bf 00 00 00 00       	mov    $0x0,%edi	8e52: R_X86_64_32	.rodata.str1.1+0x1ac3
    8e56:	e8 00 00 00 00       	call   8e5b <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x72b>	8e57: R_X86_64_PLT32	getenv-0x4
    8e5b:	48 85 c0             	test   %rax,%rax
    8e5e:	74 08                	je     8e68 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x738>
    8e60:	80 38 31             	cmpb   $0x31,(%rax)
    8e63:	0f 94 c0             	sete   %al
    8e66:	eb 02                	jmp    8e6a <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x73a>
    8e68:	31 c0                	xor    %eax,%eax
    8e6a:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 8e70 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x740>	8e6c: R_X86_64_PC32	.bss+0x8c
    8e70:	bf 00 00 00 00       	mov    $0x0,%edi	8e71: R_X86_64_32	.bss+0x98
    8e75:	e8 00 00 00 00       	call   8e7a <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x74a>	8e76: R_X86_64_PLT32	__cxa_guard_release-0x4
    8e7a:	e9 6a fb ff ff       	jmp    89e9 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x2b9>
    8e7f:	49 89 d7             	mov    %rdx,%r15
    8e82:	49 89 c6             	mov    %rax,%r14
    8e85:	bf 00 00 00 00       	mov    $0x0,%edi	8e86: R_X86_64_32	_ZGVZN4dpct7dev_mgr8instanceEvE3d_m
    8e8a:	e8 00 00 00 00       	call   8e8f <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x75f>	8e8b: R_X86_64_PLT32	__cxa_guard_abort-0x4
    8e8f:	eb 76                	jmp    8f07 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x7d7>
    8e91:	49 89 d7             	mov    %rdx,%r15
    8e94:	49 89 c6             	mov    %rax,%r14
    8e97:	48 8d 7c 24 48       	lea    0x48(%rsp),%rdi
    8e9c:	e8 00 00 00 00       	call   8ea1 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x771>	8e9d: R_X86_64_PLT32	_ZN4sycl3_V15eventD2Ev-0x4
    8ea1:	eb 2e                	jmp    8ed1 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x7a1>
    8ea3:	eb 26                	jmp    8ecb <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x79b>
    8ea5:	eb 24                	jmp    8ecb <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x79b>
    8ea7:	eb 22                	jmp    8ecb <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x79b>
    8ea9:	eb 20                	jmp    8ecb <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x79b>
    8eab:	eb 1e                	jmp    8ecb <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x79b>
    8ead:	49 89 d7             	mov    %rdx,%r15
    8eb0:	49 89 c6             	mov    %rax,%r14
    8eb3:	48 8b 3c 24          	mov    (%rsp),%rdi
    8eb7:	48 39 ef             	cmp    %rbp,%rdi
    8eba:	74 15                	je     8ed1 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x7a1>
    8ebc:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
    8ec1:	48 ff c6             	inc    %rsi
    8ec4:	e8 00 00 00 00       	call   8ec9 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x799>	8ec5: R_X86_64_PLT32	_ZdlPvm-0x4
    8ec9:	eb 06                	jmp    8ed1 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x7a1>
    8ecb:	49 89 d7             	mov    %rdx,%r15
    8ece:	49 89 c6             	mov    %rax,%r14
    8ed1:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    8ed6:	48 85 ff             	test   %rdi,%rdi
    8ed9:	0f 84 dc 00 00 00    	je     8fbb <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x88b>
    8edf:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
    8ee4:	48 29 fe             	sub    %rdi,%rsi
    8ee7:	e9 ca 00 00 00       	jmp    8fb6 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x886>
    8eec:	e9 a6 00 00 00       	jmp    8f97 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x867>
    8ef1:	49 89 d7             	mov    %rdx,%r15
    8ef4:	49 89 c6             	mov    %rax,%r14
    8ef7:	48 89 e7             	mov    %rsp,%rdi
    8efa:	e8 00 00 00 00       	call   8eff <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x7cf>	8efb: R_X86_64_PLT32	_ZN4sycl3_V16detail16PropertyListBaseD2Ev-0x4
    8eff:	eb 06                	jmp    8f07 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x7d7>
    8f01:	49 89 d7             	mov    %rdx,%r15
    8f04:	49 89 c6             	mov    %rax,%r14
    8f07:	41 83 ff 02          	cmp    $0x2,%r15d
    8f0b:	0f 85 aa 00 00 00    	jne    8fbb <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x88b>
    8f11:	4c 89 f7             	mov    %r14,%rdi
    8f14:	e8 00 00 00 00       	call   8f19 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x7e9>	8f15: R_X86_64_PLT32	__cxa_begin_catch-0x4
    8f19:	48 8b 08             	mov    (%rax),%rcx
    8f1c:	48 89 c7             	mov    %rax,%rdi
    8f1f:	ff 51 10             	call   *0x10(%rcx)
    8f22:	bf 00 00 00 00       	mov    $0x0,%edi	8f23: R_X86_64_32	_ZSt4cerr
    8f27:	48 89 c6             	mov    %rax,%rsi
    8f2a:	e8 00 00 00 00       	call   8f2f <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x7ff>	8f2b: R_X86_64_PLT32	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc-0x4
    8f2f:	48 89 c7             	mov    %rax,%rdi
    8f32:	e8 00 00 00 00       	call   8f37 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x807>	8f33: R_X86_64_PLT32	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_-0x4
    8f37:	e8 00 00 00 00       	call   8f3c <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x80c>	8f38: R_X86_64_PLT32	__cxa_end_catch-0x4
    8f3c:	41 b8 20 00 00 00    	mov    $0x20,%r8d
    8f42:	b9 00 00 00 00       	mov    $0x0,%ecx	8f43: R_X86_64_32	.rodata.str1.1+0x7ad
    8f47:	49 8b 54 24 08       	mov    0x8(%r12),%rdx
    8f4c:	45 31 f6             	xor    %r14d,%r14d
    8f4f:	4c 89 e7             	mov    %r12,%rdi
    8f52:	31 f6                	xor    %esi,%esi
    8f54:	e8 00 00 00 00       	call   8f59 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x829>	8f55: R_X86_64_PLT32	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm-0x4
    8f59:	8b 7c 24 44          	mov    0x44(%rsp),%edi
    8f5d:	85 ff                	test   %edi,%edi
    8f5f:	78 05                	js     8f66 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x836>
    8f61:	e8 6a 02 01 00       	call   191d0 <_ZN4dpctL13select_deviceEj>
    8f66:	44 89 f0             	mov    %r14d,%eax
    8f69:	48 83 c4 68          	add    $0x68,%rsp
    8f6d:	5b                   	pop    %rbx
    8f6e:	41 5c                	pop    %r12
    8f70:	41 5d                	pop    %r13
    8f72:	41 5e                	pop    %r14
    8f74:	41 5f                	pop    %r15
    8f76:	5d                   	pop    %rbp
    8f77:	c3                   	ret
    8f78:	eb 1d                	jmp    8f97 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x867>
    8f7a:	49 89 d7             	mov    %rdx,%r15
    8f7d:	49 89 c6             	mov    %rax,%r14
    8f80:	e8 00 00 00 00       	call   8f85 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x855>	8f81: R_X86_64_PLT32	__cxa_end_catch-0x4
    8f85:	eb 34                	jmp    8fbb <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x88b>
    8f87:	48 89 c7             	mov    %rax,%rdi
    8f8a:	e8 00 00 00 00       	call   8f8f <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x85f>	8f8b: R_X86_64_PLT32	__clang_call_terminate-0x4
    8f8f:	48 89 c7             	mov    %rax,%rdi
    8f92:	e8 00 00 00 00       	call   8f97 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x867>	8f93: R_X86_64_PLT32	__clang_call_terminate-0x4
    8f97:	49 89 d7             	mov    %rdx,%r15
    8f9a:	49 89 c6             	mov    %rax,%r14
    8f9d:	eb 1c                	jmp    8fbb <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x88b>
    8f9f:	49 89 d7             	mov    %rdx,%r15
    8fa2:	49 89 c6             	mov    %rax,%r14
    8fa5:	48 8b 3c 24          	mov    (%rsp),%rdi
    8fa9:	48 39 ef             	cmp    %rbp,%rdi
    8fac:	74 0d                	je     8fbb <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x88b>
    8fae:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
    8fb3:	48 ff c6             	inc    %rsi
    8fb6:	e8 00 00 00 00       	call   8fbb <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x88b>	8fb7: R_X86_64_PLT32	_ZdlPvm-0x4
    8fbb:	48 8d 7c 24 44       	lea    0x44(%rsp),%rdi
    8fc0:	e8 00 00 00 00       	call   8fc5 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x895>	8fc1: R_X86_64_PLT32	_ZN6strata4core8OnDeviceD2Ev-0x4
    8fc5:	eb 06                	jmp    8fcd <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x89d>
    8fc7:	49 89 d7             	mov    %rdx,%r15
    8fca:	49 89 c6             	mov    %rax,%r14
    8fcd:	41 83 ff 01          	cmp    $0x1,%r15d
    8fd1:	75 7d                	jne    9050 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x920>
    8fd3:	4c 89 f7             	mov    %r14,%rdi
    8fd6:	e8 00 00 00 00       	call   8fdb <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x8ab>	8fd7: R_X86_64_PLT32	__cxa_begin_catch-0x4
    8fdb:	48 89 c7             	mov    %rax,%rdi
    8fde:	e8 00 00 00 00       	call   8fe3 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x8b3>	8fdf: R_X86_64_PLT32	_ZNK4sycl3_V19exception4whatEv-0x4
    8fe3:	bf 00 00 00 00       	mov    $0x0,%edi	8fe4: R_X86_64_32	_ZSt4cerr
    8fe8:	48 89 c6             	mov    %rax,%rsi
    8feb:	e8 00 00 00 00       	call   8ff0 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x8c0>	8fec: R_X86_64_PLT32	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc-0x4
    8ff0:	48 89 c3             	mov    %rax,%rbx
    8ff3:	be 00 00 00 00       	mov    $0x0,%esi	8ff4: R_X86_64_32	.rodata.str1.1+0x581
    8ff8:	ba 19 00 00 00       	mov    $0x19,%edx
    8ffd:	48 89 c7             	mov    %rax,%rdi
    9000:	e8 00 00 00 00       	call   9005 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x8d5>	9001: R_X86_64_PLT32	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l-0x4
    9005:	be 00 00 00 00       	mov    $0x0,%esi	9006: R_X86_64_32	.rodata.str1.1+0x59b
    900a:	ba 22 00 00 00       	mov    $0x22,%edx
    900f:	48 89 df             	mov    %rbx,%rdi
    9012:	e8 00 00 00 00       	call   9017 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x8e7>	9013: R_X86_64_PLT32	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l-0x4
    9017:	be 00 00 00 00       	mov    $0x0,%esi	9018: R_X86_64_32	.rodata.str1.1+0x5be
    901c:	ba 07 00 00 00       	mov    $0x7,%edx
    9021:	48 89 df             	mov    %rbx,%rdi
    9024:	e8 00 00 00 00       	call   9029 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x8f9>	9025: R_X86_64_PLT32	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l-0x4
    9029:	48 89 df             	mov    %rbx,%rdi
    902c:	be 2a 03 00 00       	mov    $0x32a,%esi
    9031:	e8 00 00 00 00       	call   9036 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x906>	9032: R_X86_64_PLT32	_ZNSolsEi-0x4
    9036:	48 89 c7             	mov    %rax,%rdi
    9039:	e8 00 00 00 00       	call   903e <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x90e>	903a: R_X86_64_PLT32	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_-0x4
    903e:	bf 01 00 00 00       	mov    $0x1,%edi
    9043:	e8 00 00 00 00       	call   9048 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x918>	9044: R_X86_64_PLT32	exit-0x4
    9048:	49 89 c6             	mov    %rax,%r14
    904b:	e8 00 00 00 00       	call   9050 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x920>	904c: R_X86_64_PLT32	__cxa_end_catch-0x4
    9050:	4c 89 f7             	mov    %r14,%rdi
    9053:	e8 00 00 00 00       	call   9058 <_ZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_+0x928>	9054: R_X86_64_PLT32	_Unwind_Resume-0x4
    9058:	48 89 c7             	mov    %rax,%rdi
    905b:	e8 00 00 00 00       	call   9060 <_ZZN6strata4core10MtpDrafter4bindERKNS0_11WeightTableEPKNS0_10NativeHeadEPKfRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKS1_ENKUlvE0_clEv>	905c: R_X86_64_PLT32	__clang_call_terminate-0x4
