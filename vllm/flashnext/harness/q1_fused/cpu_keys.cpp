#include "key.h"
#include <iostream>
int main() {
  uint32_t bits,index;
  while(std::cin>>bits>>index) {
    const float value=std::bit_cast<float>(bits);
    std::cout<<b70_q1_packcompact_r1::mapped_key(value,index)<<'\n';
  }
}
