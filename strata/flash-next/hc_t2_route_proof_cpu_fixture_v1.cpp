#include "hc_t2_route_proof_v1.hpp"
// SYNTHETIC HOST ONLY. Fake addresses are registry identities, never GPU/event handles.
int main(int argc,char** argv){setenv("STRATA_HC_T2_ROUTE_PROOF","1",1);setenv("STRATA_VERIFY_HC_T2_REUSE","1",1);int owner=0,graph=0,foreign=0;using namespace strata::core::hc_t2_route_proof;Record r(&owner,2,1,24,48);for(int i=0;i<48;++i)r.note(true,true);r.finish();seal(&owner,&graph,2);for(int i=0;i<3;++i){bool ok=submit(&owner,(argc>1?&foreign:&graph),2,i+1);complete(&owner,&graph,ok);}close();return 0;}
