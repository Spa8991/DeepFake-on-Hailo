from hailo_sdk_client import ClientRunner

model_name = "mobilenetv4"
quantized_model_har_path = f"./hailo_conv/vHailo8L/{model_name}_quantized_norm_model.har"

runner = ClientRunner(har=quantized_model_har_path)
# By default it uses the hw_arch that is saved on the HAR. It is not recommended to change the hw_arch after Optimization.

hef = runner.compile()

file_name = f"./hailo_conv/vHailo8L/{model_name}.hef"
with open(file_name, "wb") as f:
    f.write(hef)

#profiler
har_path = f"./hailo_conv/vHailo8L/{model_name}_compiled_normv1_model.har"
runner.save_har(har_path)
#!hailo profiler {har_path}