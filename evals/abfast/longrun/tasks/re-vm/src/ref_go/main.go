package main

import (
	"bufio"
	"fmt"
	"os"
)

var partial = false

const (
	NOP = iota
	PUSHI
	PUSHC
	POP
	DUP
	SWAP
	OVER
	ROT
	ADD
	SUB
	MUL
	DIV
	MOD
	NEG
	AND
	OR
	XOR
	NOT
	SHL
	SHR
	USHR
	EQ
	NE
	LT
	LE
	GT
	GE
	JMP
	JZ
	JNZ
	CALL
	RET
	HALT
	LOAD
	STORE
	GLOAD
	GSTORE
	NEWARR
	ALOAD
	ASTORE
	ALEN
	SLEN
	SCAT
	SIDX
	SSUB
	ITOS
	STOI
	CHR
	PRINT
	PUTC
	READ
	READC
	ASSERT
	DEPTH
	PUSHL
	TYPEOF
	NOPS
)

var oplen = [NOPS]int{0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0}

type V struct {
	t int
	v int64
}
type Obj struct {
	s []byte
	a []int64
}
type Fr struct {
	loc  [16]V
	base int
	ret  uint32
}

var (
	heap   []Obj
	cells  int
	stk    [256]V
	sp     int
	glob   [64]V
	fr     [33]Fr
	nfr    int
	code   []byte
	obf    bool
	curPC  uint32
	out    = bufio.NewWriterSize(os.Stdout, 65536)
	in     = bufio.NewReader(os.Stdin)
	inv    [256]byte
	perm   [256]byte
)

func trap(rc int, msg string) {
	out.Flush()
	fmt.Fprintf(os.Stderr, "vm: %s at pc=%d\n", msg, curPC)
	os.Exit(rc)
}
func die(rc int, msg string) {
	out.Flush()
	fmt.Fprintf(os.Stderr, "vmrun: %s\n", msg)
	os.Exit(rc)
}
func initPerm() {
	st := uint32(0xB70)
	for i := 0; i < 256; i++ {
		perm[i] = byte(i)
	}
	for i := 255; i > 0; i-- {
		st = st*1103515245 + 12345
		j := int((st >> 16) % uint32(i+1))
		perm[i], perm[j] = perm[j], perm[i]
	}
	for i := 0; i < 256; i++ {
		inv[perm[i]] = byte(i)
	}
}
func csum(p []byte) uint32 {
	h := uint32(0x1505)
	for _, b := range p {
		h = (h * 33) ^ uint32(b)
	}
	return ((h << 7) | (h >> 25)) ^ 0xC0DEC0DE
}
func le32(p []byte) uint32 { return uint32(p[0]) | uint32(p[1])<<8 | uint32(p[2])<<16 | uint32(p[3])<<24 }

func dec(pc uint32, k int) byte {
	b := code[int(pc)+1+k]
	if !obf {
		return b
	}
	if k < 0 {
		return b ^ byte((pc*7+3)&0xff)
	}
	return b ^ byte((pc+uint32(13*k)+0x5A)&0xff)
}
func pop() V {
	if sp <= fr[nfr].base {
		trap(64, "stack underflow")
	}
	sp--
	return stk[sp]
}
func push(v V) {
	if sp >= 256 {
		trap(65, "stack overflow")
	}
	stk[sp] = v
	sp++
}
func I(x int64) V { return V{0, x} }
func popi() int64 {
	v := pop()
	if v.t != 0 {
		trap(66, "type error")
	}
	return v.v
}
func newobj(t int, n int) V {
	if len(heap) >= 2048 {
		trap(72, "heap exhausted")
	}
	var o Obj
	if t == 2 {
		if cells+n > 65536 {
			trap(72, "heap exhausted")
		}
		cells += n
		o.a = make([]int64, n)
	} else {
		o.s = make([]byte, n)
	}
	heap = append(heap, o)
	return V{t, int64(len(heap) - 1)}
}
func cmpv(a, b V) int {
	if a.t != b.t {
		trap(66, "type error")
	}
	if a.t == 0 {
		if a.v < b.v {
			return -1
		}
		if a.v > b.v {
			return 1
		}
		return 0
	}
	x, y := heap[a.v].s, heap[b.v].s
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	for i := 0; i < n; i++ {
		if x[i] != y[i] {
			if x[i] < y[i] {
				return -1
			}
			return 1
		}
	}
	if len(x) < len(y) {
		return -1
	}
	if len(x) > len(y) {
		return 1
	}
	return 0
}

// strtol/strtoll base 10 with C semantics (whitespace, sign, clamping); returns value and bytes consumed (0 if no digits)
func strtoC(s []byte) (int64, int) {
	i := 0
	for i < len(s) && (s[i] == ' ' || (s[i] >= 9 && s[i] <= 13)) {
		i++
	}
	neg := false
	if i < len(s) && (s[i] == '+' || s[i] == '-') {
		neg = s[i] == '-'
		i++
	}
	st := i
	var v uint64
	over := false
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		d := uint64(s[i] - '0')
		if v > (1<<63)/10 || (v == (1<<63)/10 && d > (1<<63)%10) {
			over = true
		} else {
			v = v*10 + d
		}
		i++
	}
	if i == st {
		return 0, 0
	}
	if neg {
		if over || v >= 1<<63 {
			return -1 << 63, i
		}
		return -int64(v), i
	}
	if over || v >= 1<<63 {
		return 1<<63 - 1, i
	}
	return int64(v), i
}

func main() {
	initPerm()
	budget := int64(1000000)
	info := false
	path, many := "", false
	have := false
	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "-b" {
			if i+1 >= len(args) {
				die(1, "bad budget")
			}
			i++
			v, n := strtoC([]byte(args[i]))
			if n != len(args[i]) || v <= 0 {
				die(1, "bad budget")
			}
			budget = v
		} else if a == "--info" {
			info = true
		} else if len(a) > 1 && a[0] == '-' {
			die(1, "unknown option "+a)
		} else if !have {
			path, have = a, true
		} else {
			many = true
		}
	}
	if !have || many {
		fmt.Fprintln(os.Stderr, "usage: vmrun [-b BUDGET] [--info] PROGRAM.bc")
		os.Exit(1)
	}
	buf, err := os.ReadFile(path)
	if err != nil {
		if st, e2 := os.Stat(path); e2 == nil && st.IsDir() {
			buf = nil // fopen() succeeds on a directory and reads nothing
		} else {
			die(2, "cannot open "+path)
		}
	}
	n := len(buf)
	if n < 4 || string(buf[:4]) != "BCX1" {
		die(3, "bad magic")
	}
	if n < 16 {
		die(3, "truncated file")
	}
	if buf[4] != 2 {
		die(3, fmt.Sprintf("unsupported version %d", buf[4]))
	}
	if n < 20 {
		die(3, "truncated file")
	}
	want := le32(buf[n-4:])
	if csum(buf[:n-4]) != want {
		die(4, "checksum mismatch")
	}
	flags := int(buf[5])
	obf = flags&1 != 0 && !partial
	nconst := int(buf[6]) | int(buf[7])<<8
	codelen := le32(buf[8:])
	entry := le32(buf[12:])
	p, end := 16, n-4
	nint, nstr := 0, 0
	ctype := make([]int, nconst)
	cint := make([]int64, nconst)
	cstr := make([][]byte, nconst)
	for i := 0; i < nconst; i++ {
		if p >= end {
			die(3, "malformed file")
		}
		tag := buf[p]
		p++
		if tag == 0 {
			if p+8 > end {
				die(3, "malformed file")
			}
			var v uint64
			for k := 7; k >= 0; k-- {
				v = v<<8 | uint64(buf[p+k])
			}
			p += 8
			cint[i] = int64(v)
			nint++
		} else if tag == 1 {
			if p+2 > end {
				die(3, "malformed file")
			}
			l := int(buf[p]) | int(buf[p+1])<<8
			p += 2
			if l > 4096 || p+l > end {
				die(3, "malformed file")
			}
			s := make([]byte, l)
			for j := 0; j < l; j++ {
				s[j] = buf[p+j]
				if flags&2 != 0 {
					s[j] ^= byte((0x21 + i*3 + j) & 0xff)
				}
			}
			p += l
			ctype[i] = 1
			cstr[i] = s
			nstr++
		} else {
			die(3, "bad constant tag")
		}
	}
	if uint32(end-p) != codelen {
		die(3, "malformed file")
	}
	if entry > codelen {
		die(3, "bad entry point")
	}
	code = buf[p:end]
	if info {
		fmt.Printf("version: 2\nflags: 0x%02x\nconstants: %d (%d int, %d str)\ncode: %d bytes\nentry: %d\nchecksum: %08x\n",
			flags, nconst, nint, nstr, codelen, entry, want)
		return
	}
	pc := entry
	used := int64(0)
	cl := uint32(codelen)
	for {
		curPC = pc
		if pc == cl {
			out.Flush()
			os.Exit(0)
		}
		if used >= budget {
			trap(70, "instruction budget exceeded")
		}
		used++
		rb := code[pc]
		op := int(rb)
		if obf {
			op = int(inv[rb^byte((pc*7+3)&0xff)])
		}
		if op >= NOPS {
			trap(69, "illegal instruction")
		}
		if partial {
			switch op {
			case PUSHC, CALL, RET, NEWARR, ALOAD, ASTORE, ALEN, SLEN, SCAT, SIDX, SSUB, ITOS, STOI, CHR, PUTC, READ, READC, TYPEOF:
				trap(69, "illegal instruction")
			}
		}
		if pc+1+uint32(oplen[op]) > cl {
			trap(69, "truncated instruction")
		}
		var imm int64
		npc := pc + 1 + uint32(oplen[op])
		switch {
		case op == PUSHI:
			var u uint32
			for k := 3; k >= 0; k-- {
				u = u<<8 | uint32(dec(pc, k))
			}
			imm = int64(int32(u))
		case op == PUSHL:
			var u uint64
			for k := 7; k >= 0; k-- {
				u = u<<8 | uint64(dec(pc, k))
			}
			imm = int64(u)
		case op == PUSHC || op == JMP || op == JZ || op == JNZ || op == CALL:
			u := uint32(dec(pc, 0)) | uint32(dec(pc, 1))<<8
			if op == JMP || op == JZ || op == JNZ {
				imm = int64(int16(u))
			} else {
				imm = int64(u)
			}
		case oplen[op] == 1:
			imm = int64(dec(pc, 0))
		}
		var a, b, c V
		switch op {
		case NOP:
		case PUSHI, PUSHL:
			push(I(imm))
		case PUSHC:
			if imm >= int64(nconst) {
				trap(76, "bad constant index")
			}
			if ctype[imm] == 0 {
				push(I(cint[imm]))
			} else {
				s := newobj(1, len(cstr[imm]))
				copy(heap[s.v].s, cstr[imm])
				push(s)
			}
		case POP:
			pop()
		case DUP:
			a = pop()
			push(a)
			push(a)
		case SWAP:
			b = pop()
			a = pop()
			push(b)
			push(a)
		case OVER:
			b = pop()
			a = pop()
			push(a)
			push(b)
			push(a)
		case ROT:
			c = pop()
			b = pop()
			a = pop()
			push(b)
			push(c)
			push(a)
		case ADD, SUB, MUL, DIV, MOD, AND, OR, XOR, SHL, SHR, USHR:
			if sp-fr[nfr].base < 2 {
				trap(64, "stack underflow")
			}
			b = pop()
			a = pop()
			if a.t != 0 || b.t != 0 {
				trap(66, "type error")
			}
			x, y := uint64(a.v), uint64(b.v)
			var r int64
			switch op {
			case ADD:
				r = int64(x + y)
			case SUB:
				r = int64(x - y)
			case MUL:
				r = int64(x * y)
			case DIV:
				if y == 0 {
					trap(67, "division by zero")
				}
				if a.v == -1<<63 && b.v == -1 {
					r = a.v
				} else {
					r = a.v / b.v
				}
			case MOD:
				if y == 0 {
					trap(67, "division by zero")
				}
				if b.v == -1 {
					r = 0
				} else {
					r = a.v % b.v
				}
			case AND:
				r = int64(x & y)
			case OR:
				r = int64(x | y)
			case XOR:
				r = int64(x ^ y)
			case SHL:
				r = int64(x << (y & 63))
			case SHR:
				r = a.v >> (y & 63)
			case USHR:
				r = int64(x >> (y & 63))
			}
			push(I(r))
		case NEG:
			a = pop()
			if a.t != 0 {
				trap(66, "type error")
			}
			push(I(int64(0 - uint64(a.v))))
		case NOT:
			a = pop()
			if a.t != 0 {
				trap(66, "type error")
			}
			push(I(^a.v))
		case EQ, NE, LT, LE, GT, GE:
			if sp-fr[nfr].base < 2 {
				trap(64, "stack underflow")
			}
			b = pop()
			a = pop()
			var r int
			if a.t == 2 && b.t == 2 {
				if op != EQ && op != NE {
					trap(66, "type error")
				}
				if a.v == b.v {
					r = 0
				} else {
					r = 2
				}
			} else {
				r = cmpv(a, b)
			}
			res := false
			switch op {
			case EQ:
				res = r == 0
			case NE:
				res = r != 0
			case LT:
				res = r < 0
			case LE:
				res = r <= 0
			case GT:
				res = r > 0
			case GE:
				res = r >= 0
			}
			if res {
				push(I(1))
			} else {
				push(I(0))
			}
		case JMP, JZ, JNZ:
			take := true
			if op != JMP {
				v := popi()
				if op == JZ {
					take = v == 0
				} else {
					take = v != 0
				}
			}
			if take {
				t := int64(npc) + imm
				if t < 0 || t > int64(cl) {
					trap(75, "jump out of range")
				}
				npc = uint32(t)
			}
		case CALL:
			nargs := int(dec(pc, 2))
			if nargs > 16 {
				trap(73, "bad index")
			}
			if sp-fr[nfr].base < nargs {
				trap(64, "stack underflow")
			}
			if imm >= int64(cl) {
				trap(75, "jump out of range")
			}
			if nfr >= 32 {
				trap(74, "call depth exceeded")
			}
			nf := &fr[nfr+1]
			*nf = Fr{}
			for k := 0; k < nargs; k++ {
				nf.loc[k] = stk[sp-nargs+k]
			}
			sp -= nargs
			nf.base = sp
			nf.ret = npc
			nfr++
			npc = uint32(imm)
		case RET:
			a = pop()
			if nfr == 0 {
				if a.t != 0 {
					trap(66, "type error")
				}
				out.Flush()
				os.Exit(int(a.v & 0xff))
			}
			sp = fr[nfr].base
			npc = fr[nfr].ret
			nfr--
			push(a)
		case HALT:
			v := popi()
			out.Flush()
			os.Exit(int(v & 0xff))
		case LOAD:
			if imm >= 16 {
				trap(73, "bad index")
			}
			push(fr[nfr].loc[imm])
		case STORE:
			if imm >= 16 {
				trap(73, "bad index")
			}
			fr[nfr].loc[imm] = pop()
		case GLOAD:
			if imm >= 64 {
				trap(73, "bad index")
			}
			push(glob[imm])
		case GSTORE:
			if imm >= 64 {
				trap(73, "bad index")
			}
			glob[imm] = pop()
		case NEWARR:
			k := popi()
			if k < 0 || k > 1024 {
				trap(78, "invalid array size")
			}
			push(newobj(2, int(k)))
		case ALOAD:
			i := popi()
			a = pop()
			if a.t != 2 {
				trap(66, "type error")
			}
			if i < 0 || i >= int64(len(heap[a.v].a)) {
				trap(68, "index out of range")
			}
			push(I(heap[a.v].a[i]))
		case ASTORE:
			v := popi()
			i := popi()
			a = pop()
			if a.t != 2 {
				trap(66, "type error")
			}
			if i < 0 || i >= int64(len(heap[a.v].a)) {
				trap(68, "index out of range")
			}
			heap[a.v].a[i] = v
		case ALEN:
			a = pop()
			if a.t != 2 {
				trap(66, "type error")
			}
			push(I(int64(len(heap[a.v].a))))
		case SLEN:
			a = pop()
			if a.t != 1 {
				trap(66, "type error")
			}
			push(I(int64(len(heap[a.v].s))))
		case SCAT:
			if sp-fr[nfr].base < 2 {
				trap(64, "stack underflow")
			}
			b = pop()
			a = pop()
			if a.t != 1 || b.t != 1 {
				trap(66, "type error")
			}
			l := len(heap[a.v].s) + len(heap[b.v].s)
			if l > 4096 {
				trap(77, "string too long")
			}
			r := newobj(1, l)
			copy(heap[r.v].s, heap[a.v].s)
			copy(heap[r.v].s[len(heap[a.v].s):], heap[b.v].s)
			push(r)
		case SIDX:
			i := popi()
			a = pop()
			if a.t != 1 {
				trap(66, "type error")
			}
			if i < 0 || i >= int64(len(heap[a.v].s)) {
				trap(68, "index out of range")
			}
			push(I(int64(heap[a.v].s[i])))
		case SSUB:
			l := popi()
			st := popi()
			a = pop()
			if a.t != 1 {
				trap(66, "type error")
			}
			if st < 0 || l < 0 || st+l > int64(len(heap[a.v].s)) {
				trap(68, "index out of range")
			}
			r := newobj(1, int(l))
			copy(heap[r.v].s, heap[a.v].s[st:st+l])
			push(r)
		case ITOS:
			v := popi()
			s := fmt.Sprintf("%d", v)
			r := newobj(1, len(s))
			copy(heap[r.v].s, s)
			push(r)
		case STOI:
			a = pop()
			if a.t != 1 {
				trap(66, "type error")
			}
			s := heap[a.v].s
			for i, ch := range s {
				if ch == 0 {
					s = s[:i]
					break
				}
			}
			v, _ := strtoC(s)
			push(I(v))
		case CHR:
			v := popi()
			r := newobj(1, 1)
			heap[r.v].s[0] = byte(v & 0xff)
			push(r)
		case PRINT:
			a = pop()
			if a.t == 0 {
				fmt.Fprintf(out, "%d\n", a.v)
			} else if a.t == 1 {
				out.Write(heap[a.v].s)
				out.WriteByte('\n')
			} else {
				out.WriteByte('[')
				for k, e := range heap[a.v].a {
					if k > 0 {
						out.WriteByte(',')
					}
					fmt.Fprintf(out, "%d", e)
				}
				out.WriteString("]\n")
			}
		case PUTC:
			v := popi()
			out.WriteByte(byte(v & 0xff))
		case READ:
			var line []byte
			got := false
			for {
				ch, err := in.ReadByte()
				if err != nil {
					break
				}
				got = true
				if ch == '\n' {
					break
				}
				if len(line) < 4096 {
					line = append(line, ch)
				}
			}
			if !got {
				push(I(-1))
			} else {
				r := newobj(1, len(line))
				copy(heap[r.v].s, line)
				push(r)
			}
		case READC:
			ch, err := in.ReadByte()
			if err != nil {
				push(I(-1))
			} else {
				push(I(int64(ch)))
			}
		case ASSERT:
			v := popi()
			if v == 0 {
				trap(71, "assertion failed")
			}
		case DEPTH:
			push(I(int64(sp - fr[nfr].base)))
		case TYPEOF:
			a = pop()
			push(I(int64(a.t)))
		}
		pc = npc
	}
}
