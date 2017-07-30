#ifdef _WIN32
#define COMPILER_MSVC
#endif

#include <c_api.h>
#include <iostream>

int main() {
  auto status = TF_NewStatus();
  auto graph = TF_NewGraph();
  auto sess_opts = TF_NewSessionOptions();

  constexpr char kSavedModelTagServe[] = "serve";
  const char *tags[] = {kSavedModelTagServe};

  TF_LoadSessionFromSavedModel(sess_opts, nullptr, "model", tags, 1, graph,
                               nullptr, status);

  if (TF_GetCode(status) != TF_OK) {
    std::cerr << "Unable to load session: " << TF_Message(status) << "\n";
    exit(-1);
  }

  return 0;
}
