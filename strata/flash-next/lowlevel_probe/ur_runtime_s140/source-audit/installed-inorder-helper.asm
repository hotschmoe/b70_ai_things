
/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero.so.0.12.0:     file format elf64-x86-64


Disassembly of section .text:

00000000001409c0 <_ZN21ur_platform_handle_t_23allowDriverInOrderListsEb>:
  1409c0:	f3 0f 1e fa          	endbr64
  1409c4:	41 56                	push   %r14
  1409c6:	53                   	push   %rbx
  1409c7:	50                   	push   %rax
  1409c8:	89 f3                	mov    %esi,%ebx
  1409ca:	8a 05 98 f2 09 00    	mov    0x9f298(%rip),%al        # 1dfc68 <_ZGVZN21ur_platform_handle_t_23allowDriverInOrderListsEbE27UseEnvVarDriverInOrderLists>
  1409d0:	84 c0                	test   %al,%al
  1409d2:	74 28                	je     1409fc <_ZN21ur_platform_handle_t_23allowDriverInOrderListsEb+0x3c>
  1409d4:	8a 05 9e f2 09 00    	mov    0x9f29e(%rip),%al        # 1dfc78 <_ZGVZN21ur_platform_handle_t_23allowDriverInOrderListsEbE21UseDriverInOrderLists>
  1409da:	84 c0                	test   %al,%al
  1409dc:	74 5e                	je     140a3c <_ZN21ur_platform_handle_t_23allowDriverInOrderListsEb+0x7c>
  1409de:	48 8d 05 7b f2 09 00 	lea    0x9f27b(%rip),%rax        # 1dfc60 <_ZZN21ur_platform_handle_t_23allowDriverInOrderListsEbE27UseEnvVarDriverInOrderLists>
  1409e5:	48 8d 0d 84 f2 09 00 	lea    0x9f284(%rip),%rcx        # 1dfc70 <_ZZN21ur_platform_handle_t_23allowDriverInOrderListsEbE21UseDriverInOrderLists>
  1409ec:	84 db                	test   %bl,%bl
  1409ee:	48 0f 45 c8          	cmovne %rax,%rcx
  1409f2:	8a 01                	mov    (%rcx),%al
  1409f4:	48 83 c4 08          	add    $0x8,%rsp
  1409f8:	5b                   	pop    %rbx
  1409f9:	41 5e                	pop    %r14
  1409fb:	c3                   	ret
  1409fc:	48 8d 05 65 f2 09 00 	lea    0x9f265(%rip),%rax        # 1dfc68 <_ZGVZN21ur_platform_handle_t_23allowDriverInOrderListsEbE27UseEnvVarDriverInOrderLists>
  140a03:	49 89 fe             	mov    %rdi,%r14
  140a06:	48 89 c7             	mov    %rax,%rdi
  140a09:	e8 a2 a4 09 00       	call   1daeb0 <__cxa_guard_acquire@plt>
  140a0e:	4c 89 f7             	mov    %r14,%rdi
  140a11:	85 c0                	test   %eax,%eax
  140a13:	74 bf                	je     1409d4 <_ZN21ur_platform_handle_t_23allowDriverInOrderListsEb+0x14>
  140a15:	48 8d 3d e3 5e ed ff 	lea    -0x12a11d(%rip),%rdi        # 168ff <GCC_except_table1+0x1533>
  140a1c:	e8 df a4 09 00       	call   1daf00 <getenv@plt>
  140a21:	48 85 c0             	test   %rax,%rax
  140a24:	74 60                	je     140a86 <_ZN21ur_platform_handle_t_23allowDriverInOrderListsEb+0xc6>
  140a26:	48 89 c7             	mov    %rax,%rdi
  140a29:	31 f6                	xor    %esi,%esi
  140a2b:	ba 0a 00 00 00       	mov    $0xa,%edx
  140a30:	e8 db a4 09 00       	call   1daf10 <strtol@plt>
  140a35:	85 c0                	test   %eax,%eax
  140a37:	0f 95 c0             	setne  %al
  140a3a:	eb 4c                	jmp    140a88 <_ZN21ur_platform_handle_t_23allowDriverInOrderListsEb+0xc8>
  140a3c:	48 8d 05 35 f2 09 00 	lea    0x9f235(%rip),%rax        # 1dfc78 <_ZGVZN21ur_platform_handle_t_23allowDriverInOrderListsEbE21UseDriverInOrderLists>
  140a43:	49 89 fe             	mov    %rdi,%r14
  140a46:	48 89 c7             	mov    %rax,%rdi
  140a49:	e8 62 a4 09 00       	call   1daeb0 <__cxa_guard_acquire@plt>
  140a4e:	85 c0                	test   %eax,%eax
  140a50:	74 8c                	je     1409de <_ZN21ur_platform_handle_t_23allowDriverInOrderListsEb+0x1e>
  140a52:	4c 89 f7             	mov    %r14,%rdi
  140a55:	be 01 00 00 00       	mov    $0x1,%esi
  140a5a:	ba 06 00 00 00       	mov    $0x6,%edx
  140a5f:	b9 95 7d 00 00       	mov    $0x7d95,%ecx
  140a64:	e8 e7 f6 ff ff       	call   140150 <_ZN21ur_platform_handle_t_29isDriverVersionNewerOrSimilarEjjj>
  140a69:	0a 05 f1 f1 09 00    	or     0x9f1f1(%rip),%al        # 1dfc60 <_ZZN21ur_platform_handle_t_23allowDriverInOrderListsEbE27UseEnvVarDriverInOrderLists>
  140a6f:	88 05 fb f1 09 00    	mov    %al,0x9f1fb(%rip)        # 1dfc70 <_ZZN21ur_platform_handle_t_23allowDriverInOrderListsEbE21UseDriverInOrderLists>
  140a75:	48 8d 3d fc f1 09 00 	lea    0x9f1fc(%rip),%rdi        # 1dfc78 <_ZGVZN21ur_platform_handle_t_23allowDriverInOrderListsEbE21UseDriverInOrderLists>
  140a7c:	e8 4f a4 09 00       	call   1daed0 <__cxa_guard_release@plt>
  140a81:	e9 58 ff ff ff       	jmp    1409de <_ZN21ur_platform_handle_t_23allowDriverInOrderListsEb+0x1e>
  140a86:	31 c0                	xor    %eax,%eax
  140a88:	88 05 d2 f1 09 00    	mov    %al,0x9f1d2(%rip)        # 1dfc60 <_ZZN21ur_platform_handle_t_23allowDriverInOrderListsEbE27UseEnvVarDriverInOrderLists>
  140a8e:	48 8d 3d d3 f1 09 00 	lea    0x9f1d3(%rip),%rdi        # 1dfc68 <_ZGVZN21ur_platform_handle_t_23allowDriverInOrderListsEbE27UseEnvVarDriverInOrderLists>
  140a95:	e8 36 a4 09 00       	call   1daed0 <__cxa_guard_release@plt>
  140a9a:	4c 89 f7             	mov    %r14,%rdi
  140a9d:	e9 32 ff ff ff       	jmp    1409d4 <_ZN21ur_platform_handle_t_23allowDriverInOrderListsEb+0x14>
  140aa2:	f3 0f 1e fa          	endbr64
  140aa6:	48 89 c3             	mov    %rax,%rbx
  140aa9:	48 8d 3d c8 f1 09 00 	lea    0x9f1c8(%rip),%rdi        # 1dfc78 <_ZGVZN21ur_platform_handle_t_23allowDriverInOrderListsEbE21UseDriverInOrderLists>
  140ab0:	e8 2b a4 09 00       	call   1daee0 <__cxa_guard_abort@plt>
  140ab5:	48 89 df             	mov    %rbx,%rdi
  140ab8:	e8 43 a3 09 00       	call   1dae00 <_Unwind_Resume@plt>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:

Disassembly of section .plt.sec:
00000000001b19f0 b guard variable for ur_platform_handle_t_::allowDriverInOrderLists(bool)::UseDriverInOrderLists
00000000001b19e0 b guard variable for ur_platform_handle_t_::allowDriverInOrderLists(bool)::UseEnvVarDriverInOrderLists
0000000000092c10 t ur_platform_handle_t_::allowDriverInOrderLists(bool)
0000000000147070 t ur::level_zero::urQueueCreate(ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_properties_t const*, ur_queue_handle_t_**)
00000000001473d0 t ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)
00000000001486e0 t v2::ur_queue_immediate_in_order_t::ur_queue_immediate_in_order_t(ur_context_handle_t_*, ur_device_handle_t_*, std::unique_ptr<_ze_command_list_handle_t, std::function<void (_ze_command_list_handle_t*)> >, unsigned int, unsigned int)
0000000000148450 t v2::ur_queue_immediate_in_order_t::ur_queue_immediate_in_order_t(ur_context_handle_t_*, ur_device_handle_t_*, unsigned int, _ze_command_queue_priority_t, std::optional<int>, unsigned int, unsigned int)
00000000001486e0 t v2::ur_queue_immediate_in_order_t::ur_queue_immediate_in_order_t(ur_context_handle_t_*, ur_device_handle_t_*, std::unique_ptr<_ze_command_list_handle_t, std::function<void (_ze_command_list_handle_t*)> >, unsigned int, unsigned int)
0000000000148450 t v2::ur_queue_immediate_in_order_t::ur_queue_immediate_in_order_t(ur_context_handle_t_*, ur_device_handle_t_*, unsigned int, _ze_command_queue_priority_t, std::optional<int>, unsigned int, unsigned int)
0000000000147c80 t std::_Function_base::_Base_manager<ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_manager(std::_Any_data&, std::_Any_data const&, std::_Manager_operation)
0000000000147a90 t std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)
00000000001aeed0 d typeinfo for ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0
000000000002c1d6 r typeinfo name for ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0
00000000001b19e8 b ur_platform_handle_t_::allowDriverInOrderLists(bool)::UseDriverInOrderLists
00000000001b19d8 b ur_platform_handle_t_::allowDriverInOrderLists(bool)::UseEnvVarDriverInOrderLists
