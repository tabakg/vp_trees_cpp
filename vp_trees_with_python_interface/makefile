CC = g++ -std=c++14
PYLIBPATH = $(shell python-config --exec-prefix)/lib
LIB = -L$(PYLIBPATH) $(shell python-config --libs) -lboost_python
OPTS = $(shell python-config --include) -O2

default: vp_tree.so
	@python ./python_tests.py
	@python ./python_tests_2.py
	@python ./python_tests_3.py

vp_tree.so: vp_tree.o
	$(CC) $(LIB) -Wl,-rpath,$(PYLIBPATH) -shared $< -o $@

vp_tree.o: vp_tree.cpp Makefile
	$(CC) $(OPTS) -c $< -o $@

clean:
	rm -rf *.so *.o

.PHONY: default clean
