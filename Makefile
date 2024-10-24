CC=nvcc

all: test-cuda

TARGET = deltaStepping

test-cuda:
	$(CC) $(TARGET).cu -o $(TARGET)

clean:
	rm -f $(TARGET)

