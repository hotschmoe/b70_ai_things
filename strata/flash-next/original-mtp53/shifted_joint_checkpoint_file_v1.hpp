#pragma once
// New atomic directory schema; existing per-stage SDK file schema stays unchanged.
#include "strata/kernels/original_mtp53_shifted_joint_checkpoint.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
namespace strata::kernels::original_mtp53 {
namespace shifted_joint_file53 {
namespace fs=std::filesystem;
inline void write_bytes(const fs::path& p,const std::vector<uint8_t>& bytes){int fd=::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0)throw std::runtime_error("joint exclusive file open");try{size_t at=0;while(at<bytes.size()){ssize_t n=::write(fd,bytes.data()+at,bytes.size()-at);if(n<=0)throw std::runtime_error("joint write failed");at+=size_t(n);}if(::fsync(fd))throw std::runtime_error("joint file flush failed");int rc=::close(fd);fd=-1;if(rc)throw std::runtime_error("joint file close failed");}catch(...){if(fd>=0)::close(fd);throw;}}
inline std::vector<uint8_t> read_bytes(const fs::path& p,size_t cap){struct stat before{},after{};int fd=::open(p.c_str(),O_RDONLY|O_NOFOLLOW);if(fd<0)throw std::runtime_error("joint readonly file open");try{if(::fstat(fd,&before) || !S_ISREG(before.st_mode) || before.st_nlink!=1 || before.st_size<0 || uint64_t(before.st_size)>cap)throw std::runtime_error("joint bounded regular file required");std::vector<uint8_t> b(size_t(before.st_size));size_t at=0;while(at<b.size()){ssize_t n=::read(fd,b.data()+at,b.size()-at);if(n<=0)throw std::runtime_error("joint file truncated");at+=size_t(n);}uint8_t extra;if(::read(fd,&extra,1)!=0 || ::fstat(fd,&after) || before.st_dev!=after.st_dev || before.st_ino!=after.st_ino || before.st_size!=after.st_size || before.st_mtim.tv_sec!=after.st_mtim.tv_sec || before.st_mtim.tv_nsec!=after.st_mtim.tv_nsec || before.st_ctim.tv_sec!=after.st_ctim.tv_sec || before.st_ctim.tv_nsec!=after.st_ctim.tv_nsec)throw std::runtime_error("joint opened file changed");::close(fd);return b;}catch(...){::close(fd);throw;}}
inline void save_tree(const fs::path& dir,const OriginalShiftedJointCheckpoint53& c,const std::array<strata::core::SessionFileIdentity,2>& ids){
 std::string error;size_t bytes;
 for(int i=0;i<2;++i)if(!strata::core::session_file_write((dir/("target"+std::to_string(i)+".bin")).string(),c.targets()[i],ids[i],bytes,error))throw std::runtime_error(error);
 write_bytes(dir/"original.bin",serialize_shifted53(c.original()));
 std::vector<uint8_t> previous;for(const auto& pair:c.ple_previous())for(int32_t v:pair)for(int i=0;i<4;++i)previous.push_back(uint8_t(uint32_t(v)>>(8*i)));write_bytes(dir/"ple-previous.bin",previous);
 std::string manifest="ORIGINAL-SHIFTED-JOINT58-DIR2\n"+c.digest()+"\n"+c.original().envelope_sha+"\n"+std::to_string(c.branches().size())+"\n";
 for(size_t i=0;i<c.branches().size();++i){const auto& b=c.branches()[i];manifest+=b->digest()+"\n";auto child=dir/("branch"+std::to_string(i));if(!fs::create_directory(child))throw std::runtime_error("joint exclusive branch directory");save_tree(child,*b,ids);}
 write_bytes(dir/"manifest.txt",std::vector<uint8_t>(manifest.begin(),manifest.end()));
 int directory=::open(dir.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW);if(directory<0)throw std::runtime_error("joint staging directory open failed");int rc=::fsync(directory);::close(directory);if(rc)throw std::runtime_error("joint staging directory flush failed");
}
struct Publish {bool published=false,durable=false;};
inline void save(const fs::path& destination,const OriginalShiftedJointCheckpoint53& c,const std::array<strata::core::SessionFileIdentity,2>& ids,Publish& status){
 status={};if(!destination.is_absolute() || fs::canonical(destination.parent_path())!=destination.parent_path() || fs::exists(destination))throw std::runtime_error("joint new canonical destination required");std::string pattern=(destination.parent_path()/".original-joint-owned-XXXXXX").string();std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);char* made=::mkdtemp(name.data());if(!made)throw std::runtime_error("joint exclusive owned staging directory");fs::path temp(made);
 try{save_tree(temp,c,ids);if(::syscall(SYS_renameat2,AT_FDCWD,temp.c_str(),AT_FDCWD,destination.c_str(),1))throw std::runtime_error("joint no-replace atomic publication failed");status.published=true;int fd=::open(destination.parent_path().c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW);if(fd<0)throw std::runtime_error("joint parent open failed");int rc=::fsync(fd);::close(fd);if(rc)throw std::runtime_error("joint parent flush failed after publication");status.durable=true;}catch(...){if(!status.published)fs::remove_all(temp);throw;}
}
inline std::shared_ptr<const OriginalShiftedJointCheckpoint53> read_tree(const fs::path& dir,const std::string& pinned,const std::array<strata::core::SessionFileIdentity,2>& ids,const std::array<strata::core::SessionReadLimits,2>& limits,int depth=0){
 if(depth>64 || !checkpoint_sha53(pinned) || fs::canonical(dir)!=dir)throw std::runtime_error("joint canonical tree/digest/depth required");auto raw=read_bytes(dir/"manifest.txt",8192);std::istringstream text(std::string(raw.begin(),raw.end()));std::string magic,digest,original_hash,count;std::getline(text,magic);std::getline(text,digest);std::getline(text,original_hash);std::getline(text,count);
 if(magic!="ORIGINAL-SHIFTED-JOINT58-DIR2" || digest!=pinned || !checkpoint_sha53(original_hash) || count.empty() || count.size()>2 || !std::all_of(count.begin(),count.end(),[](char c){return c>='0'&&c<='9';}))throw std::runtime_error("joint manifest schema/pin differs");int n=std::stoi(count);if(n>64)throw std::runtime_error("joint branch count bound");std::vector<std::string> child_hashes;for(int i=0;i<n;++i){std::string h;std::getline(text,h);if(!checkpoint_sha53(h))throw std::runtime_error("joint child digest invalid");child_hashes.push_back(h);}std::string extra;if(std::getline(text,extra))throw std::runtime_error("joint trailing manifest data");
 std::set<std::string> expected{"target0.bin","target1.bin","original.bin","manifest.txt","ple-previous.bin"};for(int i=0;i<n;++i)expected.insert("branch"+std::to_string(i));std::set<std::string> observed;for(const auto& e:fs::directory_iterator(dir)){if(e.is_symlink())throw std::runtime_error("joint alias entry refused");observed.insert(e.path().filename().string());}if(observed!=expected)throw std::runtime_error("joint exact file-schema roster differs");
 std::array<strata::core::SavedConversation,2> targets;std::string error;size_t bytes;
 for(int i=0;i<2;++i){const auto& l=limits[i];if(!l.geometry || !l.layer_range || l.max_tokens>4096 || l.max_checkpoints>64 || l.max_kv_layers>48 || l.max_file_bytes==UINT64_MAX || !l.admit || l.max_kv_bytes.size()<l.max_kv_layers || std::any_of(l.max_state_bytes.begin(),l.max_state_bytes.end(),[](uint64_t v){return v==UINT64_MAX;}))throw std::runtime_error("joint actual bounded target read limits/RAM admission required");if(!strata::core::session_file_read((dir/("target"+std::to_string(i)+".bin")).string(),ids[i],targets[i],bytes,error,l))throw std::runtime_error(error);}
 auto original=deserialize_shifted53(read_bytes(dir/"original.bin",41u<<20),original_hash);std::vector<std::shared_ptr<const OriginalShiftedJointCheckpoint53>> children;for(int i=0;i<n;++i)children.push_back(read_tree(dir/("branch"+std::to_string(i)),child_hashes[i],ids,limits,depth+1));auto previous_bytes=read_bytes(dir/"ple-previous.bin",16);if(previous_bytes.size()!=16)throw std::runtime_error("joint PLE metadata byte count differs");std::array<std::array<int32_t,2>,2> previous{};size_t p=0;for(auto& pair:previous)for(int32_t& v:pair){uint32_t bits=0;for(int i=0;i<4;++i)bits|=uint32_t(previous_bytes[p++])<<(8*i);std::memcpy(&v,&bits,4);}
 auto result=std::make_shared<OriginalShiftedJointCheckpoint53>(std::move(targets),std::move(original),previous,std::move(children));if(result->digest()!=pinned)throw std::runtime_error("joint full target/original bytes digest differs");return result;
}
}
}
