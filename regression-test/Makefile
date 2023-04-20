export CC_USE_MATH = 1
export CC_USE_RNG  = 1

TARGET   = regression-test
CLASSES  =
SOURCE   = $(TARGET).c $(CLASSES:%=%.c)
OBJECTS  = $(TARGET).o $(CLASSES:%=%.o)
HFILES   = $(CLASSES:%=%.h)
OPT      = -O2 -Wall
CFLAGS   = $(OPT) -I.
LDFLAGS  = -Llibnn -lnn -Llibcc -lcc -ldl -lpthread -lm
CCC      = gcc

all: $(TARGET)

$(TARGET): $(OBJECTS) libnn libcc
	$(CCC) $(OPT) $(OBJECTS) -o $@ $(LDFLAGS)

.PHONY: libnn libcc

libnn:
	$(MAKE) -C libnn

libcc:
	$(MAKE) -C libcc

clean:
	rm -f $(OBJECTS) *~ \#*\# $(TARGET)
	$(MAKE) -C libnn clean
	$(MAKE) -C libcc clean
	rm libnn libcc pcg-c-basic

$(OBJECTS): $(HFILES)
