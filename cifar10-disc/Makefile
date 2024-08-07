export CC_USE_JSMN = 1
export CC_USE_MATH = 1
export CC_USE_RNG  = 1

TARGET   = cifar10-disc
CLASSES  = libnn/cifar10/nn_cifar10              \
           libnn/cifar10-denoise/cifar10_denoise \
           cifar10_disc
SOURCE   = $(TARGET).c $(CLASSES:%=%.c)
OBJECTS  = $(TARGET).o $(CLASSES:%=%.o)
HFILES   = $(CLASSES:%=%.h)
OPT      = -O2 -Wall -Wno-format-truncation
CFLAGS   = \
	$(OPT) -I.             \
	`sdl2-config --cflags` \
	-I$(VULKAN_SDK)/include
LDFLAGS  = -Llibnn -lnn -Llibvkk -lvkk -Llibbfs -lbfs -Ltexgz -ltexgz -Llibcc -lcc -Llibsqlite3 -lsqlite3 -L$(VULKAN_SDK)/lib -lvulkan -L/usr/lib `sdl2-config --libs` -ldl -lpthread -lz -lm
CCC      = gcc

all: $(TARGET)

$(TARGET): $(OBJECTS) libbfs libcc libnn libsqlite3 libvkk texgz
	$(CCC) $(OPT) $(OBJECTS) -o $@ $(LDFLAGS)

.PHONY: libbfs libcc libnn libsqlite3 libvkk texgz

libbfs:
	$(MAKE) -C libbfs

libcc:
	$(MAKE) -C libcc

libnn:
	$(MAKE) -C libnn

libsqlite3:
	$(MAKE) -C libsqlite3

libvkk:
	$(MAKE) -C libvkk

texgz:
	$(MAKE) -C texgz

clean:
	rm -f $(OBJECTS) *~ \#*\# $(TARGET)
	$(MAKE) -C libbfs clean
	$(MAKE) -C libcc clean
	$(MAKE) -C libnn clean
	$(MAKE) -C libsqlite3 clean
	$(MAKE) -C libvkk clean
	$(MAKE) -C texgz clean
	rm jsmn libbfs libcc libnn libsqlite3 libvkk pcg-c-basic texgz

$(OBJECTS): $(HFILES)
