CXX = g++
CXXFLAGS = -g

BINS = wav-test parse-option-test apply-vad

all: $(BINS)

wav-test: wav.h
apply-vad: wav.h vad.h

.PHONY: clean
clean:
	rm -f *.o $(BINS)
