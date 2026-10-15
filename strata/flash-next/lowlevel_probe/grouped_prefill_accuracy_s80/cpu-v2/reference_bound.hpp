// S80 scalar bound contract. CPU-host only; ordinary IEEE RNE is checked by caller.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace s80ref {
struct Interval { double lo, hi; };
inline double up(double x) { return std::nextafter(x, std::numeric_limits<double>::infinity()); }
inline double down(double x) { return std::nextafter(x, -std::numeric_limits<double>::infinity()); }
inline Interval mul_nonnegative(Interval a, Interval b) {
    if (a.hi == 0 || b.hi == 0) return {0,0};
    return {std::max(0.0,down(a.lo*b.lo)),up(a.hi*b.hi)};
}
inline Interval div_positive(Interval a, Interval b) {
    if (!(a.lo >= 0 && b.lo > 0)) throw std::runtime_error("positive interval division");
    if (a.hi == 0) return {0,0};
    return {std::max(0.0,down(a.lo/b.hi)),up(a.hi/b.lo)};
}
inline Interval add_nonnegative(Interval a, Interval b) {
    if (a.hi == 0 && b.hi == 0) return {0,0};
    return {std::max(0.0,down(a.lo+b.lo)),up(a.hi+b.hi)};
}
inline Interval gamma(int n,int exponent) {
    const double unit=std::ldexp(1.0,exponent);
    const Interval nu=mul_nonnegative({double(n),double(n)},{unit,unit});
    const Interval denominator{down(1.0-nu.hi),up(1.0-nu.lo)};
    return div_positive(nu,denominator);
}
struct Bounds {double Aupper,B32,E64,total;};
inline Bounds bounds(double Ahat,Interval g32,Interval g64) {
    if (!(std::isfinite(Ahat)&&Ahat>=0)) throw std::runtime_error("invalid Ahat");
    if (Ahat==0) return {0,0,0,0}; // Exact zero, not an artificial epsilon.
    const Interval denominator{down(1.0-g64.hi),up(1.0-g64.lo)};
    const double a=div_positive({Ahat,Ahat},denominator).hi;
    const double b=mul_nonnegative(g32,{a,a}).hi;
    const double e=mul_nonnegative(g64,{a,a}).hi;
    const double total=add_nonnegative({b,b},{e,e}).hi;
    if (!(std::isfinite(a)&&std::isfinite(total))) throw std::runtime_error("bound overflow");
    return {a,b,e,total};
}
inline double residual_upper(double y,double reference) {
    if (!(std::isfinite(y)&&std::isfinite(reference))) return std::numeric_limits<double>::infinity();
    if (y==reference) return 0; // Exact subtraction of identical values.
    const double d=y-reference;
    return std::max(std::abs(down(d)),std::abs(up(d)));
}
inline bool accepts(double y,double reference,double bound) {
    if (!(std::isfinite(y)&&std::isfinite(reference)&&std::isfinite(bound)&&bound>=0)) return false;
    if (bound==0) return y==0 && reference==0; // Either signed zero allowed.
    return residual_upper(y,reference)<=bound;
}
} // namespace s80ref
