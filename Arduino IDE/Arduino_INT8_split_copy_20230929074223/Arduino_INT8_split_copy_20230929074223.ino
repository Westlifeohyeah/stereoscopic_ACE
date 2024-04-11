//#include <TensorFlowLite.h>

#include "int8_split_conv_model_inference.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"
//#include "tensorflow/lite/version.h"
const int kInputTensorSize = 16*8;
const int DIM1 = 16; // N
const int DIM2 = 8;
// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int8_t* image_data = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 300 * 1024;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace
void setup() {
  Serial.begin(115200);
  tflite::InitializeTarget();

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<10> micro_op_resolver;
  micro_op_resolver.AddStridedSlice();
  micro_op_resolver.AddConcatenation();
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddLogistic();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  // Get information about the memory area to use for the model's input.


    //tflite::AllOpsResolver micro_op_resolver;

    // Build an interpreter to run the model with.
    // NOLINTNEXTLINE(runtime-global-variables)

    // Allocate memory from the tensor_arena for the model's tensors.

  Serial.println("Model input:");
  Serial.println("input->type: " + String(input->type));
  Serial.println("dims->size: " + String(input->dims->size));
  /*for (size_t line = vert_top; line <= vert_bottom; line++) {
    for (size_t row = horz_left; row <= horz_right; row++, p++) {
      *image_data++ = tflite::FloatToQuantizedType<int8_t>(
          p[0] / 255.0f, tensor->params.scale, tensor->params.zero_point);
    }
    // move to next line
    p += ((image_width - 1) - horz_right) + horz_left;
  }*/
  if ((input->dims->size != 3) || (input->dims->data[0] != 3) ||
      (input->dims->data[1] != DIM1) ||
      (input->dims->data[2] != DIM2) ||
      (input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
    return;
  }  
  
  
  // Reshape and partition the data
  
  Serial.println("Setup Completed");
}




    // check if model loaded fine
    //if (!tf.isOk()) {
    //     Serial.print("ERROR: ");
    //    Serial.println(tf.getErrorMessage());
    //    
    //    while (true) delay(1000);
    //}

void loop() {
  
  unsigned long start_timestamp = micros();
  
  for (float i = 0; i < 100; i++) {
    

    // Get current timestamp and modulo with period
    // pick x from 0 to PI
    
    float temp2[] = {80, 80, 50, 80, 80, 50, -40, -40, 20, -40, -40, 20};
    float temp[] =
    {0.000000000000000000e+00,2.727272727272727071e-01,0.000000000000000000e+00,4.545454545454545303e-01,1.818181818181818232e-01,0.000000000000000000e+00,0.000000000000000000e+00,1.818181818181818232e-01,0.000000000000000000e+00,4.545454545454545303e-01,0.000000000000000000e+00,3.636363636363636465e-01,2.727272727272727071e-01,0.000000000000000000e+00,9.090909090909091161e-02,2.727272727272727071e-01,0.000000000000000000e+00,3.636363636363636465e-01,0.000000000000000000e+00,9.090909090909091161e-02,1.818181818181818232e-01,1.818181818181818232e-01,2.727272727272727071e-01,2.727272727272727071e-01,9.090909090909091161e-02,2.727272727272727071e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,2.727272727272727071e-01,2.727272727272727071e-01,9.090909090909091161e-02,2.727272727272727071e-01,9.090909090909091161e-02,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.818181818181818232e-01,1.818181818181818232e-01,0.000000000000000000e+00,4.545454545454545303e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,3.636363636363636465e-01,0.000000000000000000e+00,9.090909090909091161e-02,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,2.727272727272727071e-01,0.000000000000000000e+00,1.818181818181818232e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00};
    float temp1[] = {-2.349810981750490058e+01,-2.814571189880370028e+01,2.775697517395019887e+01,-3.234961700439450283e+01,-2.135284423828129974e+01,2.913620948791499998e+01};
  
    image_data = input->data.int8;
    for(int i = 0; i < kInputTensorSize; i++) {
        *image_data++ = temp[i]/input->params.scale + input->params.zero_point; //* input->params.scale + input->params.zero_point;
        Serial.print(".");
      }
    
    //input->data.f[0] = input_temp;

    uint32_t start = micros();

    if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
    }
    uint32_t timeit = micros() - start;

    Serial.print("\nIt took ");
    Serial.print(timeit);
    Serial.println(" micros to run inference");
    
    Serial.print("\t truth: ");
    for(int i = 0; i < 6; i++) {
      Serial.print(temp1[i]);

      Serial.print(" ");
    }

    Serial.print("\t predicted: ");
    for(int i = 0; i < 6; i++) {
      Serial.print(((output->data.int8[i] - output->params.zero_point) * output->params.scale) * temp2[i] + temp2[6+i]);
      Serial.print(" ");
    }

    delay(1000);
  }
}