EXEC := nvcc
FLAGS := -std=c++17
# CUDA FLAGS FOR CHOOSING ARCHITECTURE TO COMPILE THE SOURCE CODE
# Synonims, try all for seeing how it works
# CUDA_FLAGS := -arch=sm_80
# CUDA_FLAGS := -arch=compute_50 -code=compute_50,sm_50
# CUDA_FLAGS := --generate-code arch=compute_50,code=[sm_50,sm_52]
SOURCE_PATH := queryGPUInfo.cu
OUT_PATH := out/queryGPUInfo

all: build

build: 
	@if [ ! -f $(OUT_PATH) ]; then \
		mkdir -p out;\
		$(EXEC) $(FLAGS) ${CUDA_FLAGS} $(SOURCE_PATH) -o $(OUT_PATH);\
	fi

run: build
	$(OUT_PATH)

clean:
	rm -f $(OUT_PATH)