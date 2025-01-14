# Basic vector addition

Compares the time taken for completing the vector addition operations with N elements. Compared cases: CPU, CPU+Parallelism, CUDA

run with: <b>make run vec_size=N<b>

runtimes (in microseconds):
|N|CPU|CPU + Parallelism|CUDA|
|--|:---:|:----:|:---:|
|10000|84|1158|171|
|100000|437|1225|141|
|1000000|3885|2418|267|
|10000000|37619|15267|263|
|100000000|425410|318696|260|





