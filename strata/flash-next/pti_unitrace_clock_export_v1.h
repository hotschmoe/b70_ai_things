// Default-off output-only metadata for the pinned PTI fixture experiment.
#ifndef B70_PTI_UNITRACE_CLOCK_EXPORT_V1_H
#define B70_PTI_UNITRACE_CLOCK_EXPORT_V1_H
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <locale>
#include <sstream>
#include <string>
#include <utility>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace b70_pti_clock_v1 {
inline bool Enabled() {
  const char* value = std::getenv("UNITRACE_CLOCK_METADATA_PATH");
  return value != nullptr && value[0] != '\0';
}
[[noreturn]] inline void Fail(const char* reason) {
  std::fprintf(stderr, "[B70_CLOCK_EXPORT_ERROR] %s\n", reason);
  std::fflush(stderr);
  _exit(86);
}
inline uint64_t DoubleBits(double value) {
  static_assert(sizeof(double) == sizeof(uint64_t), "exact F64 payload required");
  uint64_t bits; std::memcpy(&bits, &value, sizeof(bits)); return bits;
}
inline uint64_t UuidWord(const void* value) {
  uint64_t word; std::memcpy(&word, value, sizeof(word)); return word;
}
struct Writer {
  std::mutex lock;
  int fd = -1;
  pid_t owner = 0;
  uint64_t sequence = 0;
  uint64_t bytes = 0;
  ~Writer() { if (fd >= 0 && ::close(fd) != 0) Fail("final close failed"); }
  void Emit(const char* kind, std::initializer_list<std::pair<const char*,uint64_t>> fields) {
    // OFF performs no allocation, lock, file operation or extra clock query.
    if (!Enabled()) return;
    std::lock_guard<std::mutex> held(lock);
    const pid_t pid = ::getpid();
    if (fd < 0) {
      const char* supplied = std::getenv("UNITRACE_CLOCK_METADATA_PATH");
      if (supplied == nullptr || std::strcmp(supplied, "/results/unitrace/clock.{pid}.jsonl") != 0)
        Fail("exact declared per-process output template required");
      const std::string path = std::string("/results/unitrace/clock.") + std::to_string(pid) + ".jsonl";
      fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
      if (fd < 0) Fail("exclusive output open failed");
      struct stat info;
      if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_nlink != 1 || info.st_uid != ::geteuid())
        Fail("regular owned output required");
      owner = pid;
    }
    if (owner != pid) Fail("inherited writer owner changed");
    std::ostringstream row;
    row.imbue(std::locale::classic());
    row << "{\"schema\":1,\"namespace\":\"b70-pti-clock-export-v1\",\"source_base\":\"6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033\",\"pid\":" << pid
        << ",\"writer_tid\":" << ::syscall(SYS_gettid) << ",\"sequence\":" << ++sequence << ",\"kind\":\"" << kind << "\"";
    for (const auto& field : fields) row << ",\"" << field.first << "\":" << field.second;
    row << "}\n";
    const std::string value = row.str();
    if (value.size() > 4096 || bytes + value.size() > (8u << 20)) Fail("metadata output bound exceeded");
    size_t offset = 0;
    while (offset < value.size()) {
      const ssize_t done = ::write(fd, value.data() + offset, value.size() - offset);
      if (done < 0 && errno == EINTR) continue;
      if (done <= 0) Fail("metadata output write failed");
      offset += static_cast<size_t>(done);
    }
    bytes += value.size();
  }
};
inline void Emit(const char* kind, std::initializer_list<std::pair<const char*,uint64_t>> fields) {
  if (!Enabled()) return;
  static Writer writer;
  writer.Emit(kind, fields);
}
} // namespace b70_pti_clock_v1
#endif
