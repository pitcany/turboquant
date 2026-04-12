# Build the Stage-2 QJL CPU reference as a shared library for ctypes tests.
# When integrating into ggml-tq3, these sources go into ggml/src/ and the
# library is built by ggml's CMake — this Makefile is sandbox-only.

CFLAGS ?= -O2 -Wall -Wextra -fPIC

libggml_tq_paper.so: ggml-tq-paper.c ggml-tq-paper.h tqp_constants.pt tqp_constants_d128.h tqp_constants_d256.h tqp_centroids_d128.h tqp_centroids_d256.h
	$(CC) $(CFLAGS) -shared -o $@ ggml-tq-paper.c

tqp_constants.pt tqp_constants_d128.h tqp_constants_d256.h tqp_centroids_d128.h tqp_centroids_d256.h:
	cd ../python && python3 generate_constants.py

.PHONY: clean test
clean:
	rm -f libggml_tq_paper.so *.o

test: libggml_tq_paper.so
	cd ../python && python3 -m pytest test_tq_paper.py test_c_vs_python.py -v
