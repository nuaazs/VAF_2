// Girotto Pietro - Matr. 1216355
#include "vad.hpp"    
#include <complex>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

VAD::VAD(int fileNum) {
  fileNumber = fileNum;
  inputPath = "inputData/inputaudio" + to_string(fileNumber) + ".data";
  lastVoice = 0;
}

VAD::VAD(string path) { 
  fileNumber = 0;
  inputPath = path;
  lastVoice = 0;
}
void VAD::processData() {
  string outputFileName = "outputData/outputVAD" + to_string(fileNumber) + ".data";
  string txtFileName = "outputTXT/ouputVAD" + to_string(fileNumber) + ".txt";

  ifstream inputStream(inputPath, ifstream::binary);
  ofstream outputStream(outputFileName, ofstream::binary);
  ofstream txt(txtFileName);

  vector<signed char> lastSent;

  // starting processing
  while(inputStream.good()) {
    vector<signed char> packet;
    //reading from input
    for(int i = 0; i < PACKET_SIZE && inputStream.good(); i++) {
      signed char sample;
      inputStream.read((char*) &sample, sizeof(sample));
      packet.push_back(sample);
    }

    if(isVoice(packet)) {
      // writing out any previous silences before writing current packet
      if(!lastSent.empty()) {
        for(int i = 0; i < lastSent.size(); i++) {
          outputStream.write((char *) &lastSent[i], sizeof(lastSent[i]));
        }
        txt.put('-');
      }
      
      // writing the current packet
      for(int i = 0; i < PACKET_SIZE; i++) {
        outputStream.write((char *) &packet[i], sizeof(packet[i]));
      }
      txt.put('1');

      lastSent.clear();
    } else {
      // appending the packet to previous silence
      lastSent.insert(lastSent.end(), packet.begin(), packet.end());
    }
  }

  // closing file streams
  inputStream.close();

  // writing any unsent silences at end of file
  if(!lastSent.empty()) {
    for(int i = 0; i < lastSent.size(); i++) {
      outputStream.write((char *) &lastSent[i], sizeof(lastSent[i]));
    }
    txt.put('-');
  }

  outputStream.close();
  txt.close();

  return;
}


bool VAD::isVoice(vector<signed char> &packet) {

  vector<complex<double>> complexValues;
  for(int i = 0; i < packet.size(); i++) {
    complex<double> c((double) packet[i], 0.0);
    complexValues.push_back(c);
  }
  vector<complex<double>> fftOutput = fft(complexValues);

  // Finding the greatest magnitude
  double maxMagnitude = -1;
  for(int i = 0; i < (fftOutput.size() / 2); i++) {
    double magnitude = sqrt((fftOutput[i].real() * fftOutput[i].real()) + (fftOutput[i].imag() * fftOutput[i].imag()));
    maxMagnitude = max(magnitude, maxMagnitude);
  }

  double peak = maxMagnitude;

  if(peak <= maxFrequency && peak >= minFrequency) {
    // Next iteration will be recorded regardless of their content
    lastVoice = 2;
    return true;
  }

  // Recently found voice-related packet
  if(lastVoice > 0) {
    lastVoice--;
    return true;
  }

  // Packet does not match
  return false;
}

vector<complex<double>> VAD::fft(vector<complex<double>> sample) {

  if(sample.size() == 1) return sample;

  vector<complex<double>> even;
  for(int i = 0; i < (sample.size() / 2); i++) {
    complex<double> c(sample[i*2]);
    even.push_back(c);
  }
  even = fft(even);

  vector<complex<double>> odd;
  for(int i = 0; i < (sample.size() / 2); i++) {
    complex<double> c(sample[i*2 + 1]);
    odd.push_back(c);
  }
  odd = fft(odd);

  vector<complex<double>> res(sample.size());

  for(int i = 0; i < (sample.size() / 2); i++) {
    double s = (i * PI * 2.0) / sample.size();
    
    complex<double> sk(cos(s), sin(s));
    complex<double> evenNum = even[i];
    complex<double> oddNum = odd[i];

    res[i] = evenNum + (sk * oddNum);
    res[i + (sample.size() / 2)] = evenNum - (sk * oddNum);
  }

  return res;
}