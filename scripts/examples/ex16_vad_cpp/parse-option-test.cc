/* Created on 2016-09-14
 * Author: Binbin Zhang
 */

#include "parse-option.h"


int main(int argc, char *argv[]) {
    const char *usage = "Simple parse option test file\n";
    ParseOptions option(usage);

    bool flag = false;
    option.Register("flag", &flag, "if use flag");
    int port = 8000;
    option.Register("port", &port, "bind port for tcp");
    std::string ip = "127.0.0.1";
    option.Register("ip", &ip, "server ip");
    float thresh = 1000.0;
    option.Register("thresh", &thresh, "threshold for vad");
   
    option.Read(argc, argv);

    if (option.NumArgs() != 2) {
        option.PrintUsage();
        exit(1);
    }

    std::cout << "flag " << flag << "\n";
    std::cout << "port " << port << "\n";
    std::cout << "ip " << ip << "\n";
    std::cout << "thresh " << thresh << "\n";

    std::cout << option.GetArg(1) << " "
              << option.GetArg(2) << "\n";
}



