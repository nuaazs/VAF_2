# 微服务管理工具

这是一个用于管理微服务的Shell脚本工具。它可以帮助你启动、停止、查看状态和测试各个微服务。

## 功能

此工具支持以下功能：

- 启动指定的微服务
- 停止指定的微服务
- 查看指定微服务的运行状态
- 测试指定微服务的功能

## 使用方法

1. 将`vaf_manager.sh`脚本文件复制到你的项目目录中。

2. 打开终端，并导航到项目目录。

3. 使用以下命令来操作微服务：

   - 启动服务：`./vaf_manager.sh start <service_name>` 或 `./vaf_manager.sh start all`
  
     例如，启动名为`asr_server`的微服务：
  
     ```
     ./vaf_manager.sh start asr_server
     ```
  
     或者，启动所有微服务：
  
     ```
     ./vaf_manager.sh start all
     ```

   - 停止服务：`./vaf_manager.sh stop <service_name>` 或 `./vaf_manager.sh stop all`
  
     例如，停止名为`asr_server`的微服务：
  
     ```
     ./vaf_manager.sh stop asr_server
     ```
  
     或者，停止所有微服务：
  
     ```
     ./vaf_manager.sh stop all
     ```

   - 查看服务状态：`./vaf_manager.sh status <service_name>` 或 `./vaf_manager.sh status all`
  
     例如，查看名为`asr_server`的微服务状态：
  
     ```
     ./vaf_manager.sh status asr_server
     ```
  
     或者，查看所有微服务的状态：
  
     ```
     ./vaf_manager.sh status all
     ```

   - 测试服务：`./vaf_manager.sh test <service_name>` 或 `./vaf_manager.sh test all`
  
     例如，测试名为`asr_server`的微服务：
  
     ```
     ./vaf_manager.sh test asr_server
     ```
  
     或者，测试所有微服务：
  
     ```
     ./vaf_manager.sh test all
     ```

4. 根据需要选择合适的命令来管理和测试你的微服务。

请确保在运行测试命令之前，每个微服务目录中都有一个名为"test_api.py"的测试脚本文件。

## 配置

如果需要修改微服务的路径、端口号或工作进程数，可以在脚本中进行配置。根据你的项目需求，修改以下配置变量的值：

```bash
export MICRO_SERVICE_PATH="/home/zhaosheng/asr_damo_websocket/online/microservice"

declare -A server_paths=(
    ["asr_server"]="${MICRO_SERVICE_PATH}/servers/asr_server/offline/python_cpu"
    ["encode_server"]="${MICRO_SERVICE_PATH}/servers/encode_server"
    ["language_classify_server"]="${MICRO_SERVICE_PATH}/servers/language_classify_server"
    ["text_classify_server"]="${MICRO_SERVICE_PATH}/servers/text_classify_server"
    ["vad_server_nn"]="${MICRO_SERVICE_PATH}/servers/vad_server/nn"
    ["vad_server_energybase"]="${MICRO_SERVICE_PATH}/servers/vad_server/energybase"
)

declare -A ports=(
    ["asr_server"]="5000"
    ["encode_server"]="5001"
    ["language_classify_server"]="5002"
    ["text_classify_server"]="5003"
    ["vad_server_nn"]="5004"
    ["vad_server_energybase"]="5005"
)

declare -A workers=(
    ["asr_server"]="1"
    ["encode_server"]="1"
    ["language_classify_server"]="1"
    ["text_classify_server"]="1"
    ["vad_server_nn"]="1"
    ["vad_server_energybase"]="1"
)
```

根据你的实际情况修改以上配置变量的值，并保存更改。

## 注意事项

- 请确保在运行脚本之前，使用`chmod`命令为脚本文件添加可执行权限：

  ```
  chmod +x vaf_manager.sh
  ```

- 在调用各个微服务之前，请确保每个微服务目录中都有一个名为"test_api.py"的测试脚本文件。

- 如果需要查看更详细的日志信息，请查看各个微服务目录下的日志文件。

- 请确保在使用该工具时遵守相关法律法规，并不得违法利用微服务。

-------------------------------------------------------------------------------------------------------------------

希望这个README教程对你有所帮助！如果有任何疑问，请随时提问。