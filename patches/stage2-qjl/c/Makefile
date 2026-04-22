# Build the Stage-2 QJL CPU reference as a shared library for ctypes tests.
# When integrating into ggml-tq3, these sources go into ggml/src/ and the
# library is built by ggml's CMake — this Makefile is sandbox-only.

CFLAGS ?= -O2 -Wall -Wextra -fPIC

GENERATED = tqp_constants.pt \
            tqp_constants_d64.h tqp_constants_d128.h tqp_constants_d256.h \
            tqp_centroids_d64.h tqp_centroids_d128.h tqp_centroids_d256.h \
            tqp_centroids_d64_b2.h tqp_centroids_d64_b3.h tqp_centroids_d64_b4.h \
            tqp_centroids_d128_b2.h tqp_centroids_d128_b3.h tqp_centroids_d128_b4.h \
            tqp_centroids_d256_b2.h tqp_centroids_d256_b3.h tqp_centroids_d256_b4.h

libggml_tq_paper.so: ggml-tq-paper.c ggml-tq-paper.h $(GENERATED)
	$(CC) $(CFLAGS) -shared -o $@ ggml-tq-paper.c -lm -lpthread

$(GENERATED):
	cd ../python && python3 generate_constants.py --bits 2,3,4 --dims 64,128,256

.PHONY: clean test
clean:
	rm -f libggml_tq_paper.so *.o

test: libggml_tq_paper.so
	cd ../python && python3 -m pytest test_tq_paper.py test_c_vs_python.py -v
