# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/zhaosheng/asr_damo_websocket/cpp/FunASR/funasr/runtime/websocket/third_party/websocket"
  "/home/zhaosheng/asr_damo_websocket/cpp/FunASR/funasr/runtime/websocket/build/_deps/websocketpp-build"
  "/home/zhaosheng/asr_damo_websocket/cpp/FunASR/funasr/runtime/websocket/build/_deps/websocketpp-subbuild/websocketpp-populate-prefix"
  "/home/zhaosheng/asr_damo_websocket/cpp/FunASR/funasr/runtime/websocket/build/_deps/websocketpp-subbuild/websocketpp-populate-prefix/tmp"
  "/home/zhaosheng/asr_damo_websocket/cpp/FunASR/funasr/runtime/websocket/build/_deps/websocketpp-subbuild/websocketpp-populate-prefix/src/websocketpp-populate-stamp"
  "/home/zhaosheng/asr_damo_websocket/cpp/FunASR/funasr/runtime/websocket/build/_deps/websocketpp-subbuild/websocketpp-populate-prefix/src"
  "/home/zhaosheng/asr_damo_websocket/cpp/FunASR/funasr/runtime/websocket/build/_deps/websocketpp-subbuild/websocketpp-populate-prefix/src/websocketpp-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/zhaosheng/asr_damo_websocket/cpp/FunASR/funasr/runtime/websocket/build/_deps/websocketpp-subbuild/websocketpp-populate-prefix/src/websocketpp-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/zhaosheng/asr_damo_websocket/cpp/FunASR/funasr/runtime/websocket/build/_deps/websocketpp-subbuild/websocketpp-populate-prefix/src/websocketpp-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
