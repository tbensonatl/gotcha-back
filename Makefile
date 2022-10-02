all: release

.PHONY:
release:
	mkdir -p build-release
	cd build-release && cmake -DCMAKE_BUILD_TYPE=RELEASE .. && make

.PHONY:
debug:
	mkdir -p build-debug
	cd build-debug && cmake -DCMAKE_BUILD_TYPE=DEBUG .. && make

clean:
	rm -rf build-release build-debug
