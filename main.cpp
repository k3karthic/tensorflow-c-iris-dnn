#ifdef _WIN32
#define COMPILER_MSVC
#endif

#include <iostream>
#include <memory>
#include <tensorflow/c_api.h>

TF_Tensor *FloatTensor(const int64_t *dims, int num_dims, const float *values) {
  int64_t num_values = 1;

  for (int i = 0; i < num_dims; ++i) {
    num_values *= dims[i];
  }

  TF_Tensor *t =
      TF_AllocateTensor(TF_FLOAT, dims, num_dims, sizeof(float) * num_values);

  memcpy(TF_TensorData(t), values, sizeof(float) * num_values);

  return t;
}

int main() {
  auto status = TF_NewStatus();
  auto graph = TF_NewGraph();
  auto sess_opts = TF_NewSessionOptions();

  constexpr char kSavedModelTagServe[] = "serve";
  const char *tags[] = {kSavedModelTagServe};

  auto session = TF_LoadSessionFromSavedModel(sess_opts, nullptr, "model", tags,
                                              1, graph, nullptr, status);

  if (TF_GetCode(status) != TF_OK) {
    std::cerr << "Unable to load session: " << TF_Message(status) << "\n";
    exit(-1);
  }

  TF_DeleteSessionOptions(sess_opts);

  const int ninputs = 1;
  const int noutputs = 1;

  std::unique_ptr<TF_Output[]> inputs(new TF_Output[ninputs]);
  std::unique_ptr<TF_Tensor *[]> input_values(new TF_Tensor *[ninputs]);
  std::unique_ptr<TF_Output[]> outputs(new TF_Output[noutputs]);
  std::unique_ptr<TF_Tensor *[]> output_values(new TF_Tensor *[noutputs]);

  const float input_row[4] = {6.4f, 3.2f, 4.5f, 1.5f};
  const int64_t input_dims[2] = {1, 4};
  auto input_tensor =
      FloatTensor((const int64_t *)&input_dims, 2, (const float *)&input_row);

  inputs.get()[0] = TF_Output{TF_GraphOperationByName(graph, "InputData/X"), 0};

  if (TF_GetCode(status) != TF_OK) {
    std::cerr << "Unable to fetch input operation: " << TF_Message(status)
              << "\n";
    exit(-1);
  }

  input_values.get()[0] = input_tensor;

  const float output_row[3] = {0, 0, 0};
  const int64_t output_dims[2] = {1, 3};
  auto output_tensor =
      FloatTensor((const int64_t *)&output_dims, 2, (const float *)&output_row);

  outputs.get()[0] = TF_Output{TF_GraphOperationByName(graph, "Top_3"), 0};

  output_values.get()[0] = output_tensor;

  TF_SessionRun(session, nullptr, inputs.get(), input_values.get(), ninputs,
                outputs.get(), output_values.get(), noutputs, nullptr, 0,
                nullptr, status);

  if (TF_GetCode(status) != TF_OK) {
    std::cerr << "Unable to run session: " << TF_Message(status) << "\n";
    exit(-1);
  }

  float *values = static_cast<float *>(TF_TensorData(output_values.get()[0]));
  for (int i = 0; i < 3; i++) {
    std::cout << values[i] << "\n";
  }

  TF_DeleteTensor(input_tensor);
  TF_DeleteTensor(output_tensor);

  return 0;
}
