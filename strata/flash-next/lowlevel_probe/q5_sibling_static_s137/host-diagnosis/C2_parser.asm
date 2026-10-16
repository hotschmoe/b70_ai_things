0000000000072660 <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv>:
   72660:	41 56                	push   %r14
   72662:	53                   	push   %rbx
   72663:	50                   	push   %rax
   72664:	bf 00 00 00 00       	mov    $0x0,%edi	72665: R_X86_64_32	.rodata.str1.1+0x29eb
   72669:	e8 00 00 00 00       	call   7266e <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv+0xe>	7266a: R_X86_64_PLT32	getenv-0x4
   7266e:	48 89 c7             	mov    %rax,%rdi
   72671:	31 c0                	xor    %eax,%eax
   72673:	48 85 ff             	test   %rdi,%rdi
   72676:	74 3b                	je     726b3 <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv+0x53>
   72678:	80 3f 30             	cmpb   $0x30,(%rdi)
   7267b:	75 06                	jne    72683 <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv+0x23>
   7267d:	80 7f 01 00          	cmpb   $0x0,0x1(%rdi)
   72681:	74 30                	je     726b3 <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv+0x53>
   72683:	48 c7 04 24 00 00 00 00 	movq   $0x0,(%rsp)
   7268b:	48 89 e6             	mov    %rsp,%rsi
   7268e:	ba 10 00 00 00       	mov    $0x10,%edx
   72693:	e8 00 00 00 00       	call   72698 <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv+0x38>	72694: R_X86_64_PLT32	__isoc23_strtoul-0x4
   72698:	48 8b 0c 24          	mov    (%rsp),%rcx
   7269c:	48 85 c9             	test   %rcx,%rcx
   7269f:	74 1a                	je     726bb <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv+0x5b>
   726a1:	80 39 00             	cmpb   $0x0,(%rcx)
   726a4:	75 15                	jne    726bb <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv+0x5b>
   726a6:	48 3d 01 01 01 01    	cmp    $0x1010101,%rax
   726ac:	75 0d                	jne    726bb <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv+0x5b>
   726ae:	b8 01 01 01 01       	mov    $0x1010101,%eax
   726b3:	48 83 c4 08          	add    $0x8,%rsp
   726b7:	5b                   	pop    %rbx
   726b8:	41 5e                	pop    %r14
   726ba:	c3                   	ret
   726bb:	bf 10 00 00 00       	mov    $0x10,%edi
   726c0:	e8 00 00 00 00       	call   726c5 <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv+0x65>	726c1: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   726c5:	48 89 c3             	mov    %rax,%rbx
   726c8:	be 00 00 00 00       	mov    $0x0,%esi	726c9: R_X86_64_32	.rodata.str1.1+0x2a07
   726cd:	48 89 c7             	mov    %rax,%rdi
   726d0:	e8 00 00 00 00       	call   726d5 <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv+0x75>	726d1: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   726d5:	be 00 00 00 00       	mov    $0x0,%esi	726d6: R_X86_64_32	_ZTISt16invalid_argument
   726da:	ba 00 00 00 00       	mov    $0x0,%edx	726db: R_X86_64_32	_ZNSt16invalid_argumentD1Ev
   726df:	48 89 df             	mov    %rbx,%rdi
   726e2:	e8 00 00 00 00       	call   726e7 <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv+0x87>	726e3: R_X86_64_PLT32	__cxa_throw-0x4
   726e7:	49 89 c6             	mov    %rax,%r14
   726ea:	48 89 df             	mov    %rbx,%rdi
   726ed:	e8 00 00 00 00       	call   726f2 <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv+0x92>	726ee: R_X86_64_PLT32	__cxa_free_exception-0x4
   726f2:	4c 89 f7             	mov    %r14,%rdi
   726f5:	e8 00 00 00 00       	call   726fa <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv+0x9a>	726f6: R_X86_64_PLT32	_Unwind_Resume-0x4
