EXEC := nvcc
FLAGS := -std=c++17 -ltbb
SOURCE_PATH := simpleVectorAdd.cu
OUT_PATH := out/simpleVectorAdd

all: build

build: simpleVectorAdd

simpleVectorAdd:
	@if [ ! -f $(OUT_PATH) ]; then \
		mkdir -p out;\
		$(EXEC) $(FLAGS) $(SOURCE_PATH) -o $(OUT_PATH);\
	fi

run: simpleVectorAdd
	$(OUT_PATH) $(vec_size)

clean:
	rm -f $(OUT_PATH)