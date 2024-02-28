// Modified on 2023-06-17 by Sheng Zhao

#include "wav.h"


int main(int argc, char *argv[]) {

    const char *usage = "Test wave reader and writer\n"
                        "Usage: wav-test wav_in_file wav_output_file\n";
    if (argc != 3) {
        printf(usage);
        exit(-1);
    }

    WaveReader reader(argv[1], 1, 16);

    WaveWriter writer(reader.Data(), reader.NumSample(), reader.NumChannel(),
                     reader.BitDepth());
    writer.Write(argv[2]);
    return 0;
}
