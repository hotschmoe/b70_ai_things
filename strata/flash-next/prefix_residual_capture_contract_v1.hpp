#pragma once
#include <cstddef>
#include <cstring>
#include <stdexcept>
namespace strata::core::prefix_residual30 {
constexpr int max_rows=8, layers=48, phases=3, row_floats=4*2560;
constexpr std::size_t max_data_bytes=std::size_t(layers)*phases*max_rows*row_floats*sizeof(float);
static_assert(max_data_bytes==47185920);
constexpr std::size_t aggregate_diagnostic_budget=128u*1024u*1024u;
// Two P30 owners plus bounded prior SFD/L0 observers; actual allocation receipts still required.
static_assert(2*(max_data_bytes+32)+16u*1024u*1024u<aggregate_diagnostic_budget);
inline std::size_t stage_bytes(int lb,int le){if(lb<0||lb>=le||le>layers)throw std::invalid_argument("P30 stagebounds");return std::size_t(le-lb)*phases*max_rows*row_floats*sizeof(float);}
inline std::size_t extent(int rows){if(rows<1||rows>max_rows)throw std::invalid_argument("P30 rows outside1..8");return std::size_t(rows)*row_floats*sizeof(float);}
inline int phase_index(const char* phase){if(!phase)throw std::invalid_argument("P30 phase absent");return !std::strcmp(phase,"input")?0:!std::strcmp(phase,"attention")?1:!std::strcmp(phase,"ffn")?2:throw std::invalid_argument("P30 phase differs");}
}
