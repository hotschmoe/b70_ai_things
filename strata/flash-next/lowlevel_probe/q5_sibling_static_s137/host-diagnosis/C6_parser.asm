00000000000727c0 <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv>:
   727c0:	41 56                	push   %r14
   727c2:	53                   	push   %rbx
   727c3:	50                   	push   %rax
   727c4:	bf 00 00 00 00       	mov    $0x0,%edi	727c5: R_X86_64_32	.rodata.str1.1+0x2b3d
   727c9:	e8 00 00 00 00       	call   727ce <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv+0xe>	727ca: R_X86_64_PLT32	getenv-0x4
   727ce:	48 89 c7             	mov    %rax,%rdi
   727d1:	31 c0                	xor    %eax,%eax
   727d3:	48 85 ff             	test   %rdi,%rdi
   727d6:	74 3b                	je     72813 <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv+0x53>
   727d8:	80 3f 30             	cmpb   $0x30,(%rdi)
   727db:	75 06                	jne    727e3 <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv+0x23>
   727dd:	80 7f 01 00          	cmpb   $0x0,0x1(%rdi)
   727e1:	74 30                	je     72813 <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv+0x53>
   727e3:	48 c7 04 24 00 00 00 00 	movq   $0x0,(%rsp)
   727eb:	48 89 e6             	mov    %rsp,%rsi
   727ee:	ba 10 00 00 00       	mov    $0x10,%edx
   727f3:	e8 00 00 00 00       	call   727f8 <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv+0x38>	727f4: R_X86_64_PLT32	__isoc23_strtoul-0x4
   727f8:	48 8b 0c 24          	mov    (%rsp),%rcx
   727fc:	48 85 c9             	test   %rcx,%rcx
   727ff:	74 1a                	je     7281b <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv+0x5b>
   72801:	80 39 00             	cmpb   $0x0,(%rcx)
   72804:	75 15                	jne    7281b <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv+0x5b>
   72806:	48 3d 01 01 01 01    	cmp    $0x1010101,%rax
   7280c:	75 0d                	jne    7281b <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv+0x5b>
   7280e:	b8 01 01 01 01       	mov    $0x1010101,%eax
   72813:	48 83 c4 08          	add    $0x8,%rsp
   72817:	5b                   	pop    %rbx
   72818:	41 5e                	pop    %r14
   7281a:	c3                   	ret
   7281b:	bf 10 00 00 00       	mov    $0x10,%edi
   72820:	e8 00 00 00 00       	call   72825 <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv+0x65>	72821: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   72825:	48 89 c3             	mov    %rax,%rbx
   72828:	be 00 00 00 00       	mov    $0x0,%esi	72829: R_X86_64_32	.rodata.str1.1+0x2b5c
   7282d:	48 89 c7             	mov    %rax,%rdi
   72830:	e8 00 00 00 00       	call   72835 <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv+0x75>	72831: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   72835:	be 00 00 00 00       	mov    $0x0,%esi	72836: R_X86_64_32	_ZTISt16invalid_argument
   7283a:	ba 00 00 00 00       	mov    $0x0,%edx	7283b: R_X86_64_32	_ZNSt16invalid_argumentD1Ev
   7283f:	48 89 df             	mov    %rbx,%rdi
   72842:	e8 00 00 00 00       	call   72847 <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv+0x87>	72843: R_X86_64_PLT32	__cxa_throw-0x4
   72847:	49 89 c6             	mov    %rax,%r14
   7284a:	48 89 df             	mov    %rbx,%rdi
   7284d:	e8 00 00 00 00       	call   72852 <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv+0x92>	7284e: R_X86_64_PLT32	__cxa_free_exception-0x4
   72852:	4c 89 f7             	mov    %r14,%rdi
   72855:	e8 00 00 00 00       	call   7285a <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv+0x9a>	72856: R_X86_64_PLT32	_Unwind_Resume-0x4
