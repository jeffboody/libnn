export CC_USE_MATH = 1
export CC_USE_RNG  = 1

TARGET   = cnn-test
CLASSES  =
SOURCE   = $(TARGET).c $(CLASSES:%=%.c)
OBJECTS  = $(TARGET).o $(CLASSES:%=%.o)
HFILES   = $(CLASSES:%=%.h)
OPT      = -O2 -Wall
CFLAGS   = $(OPT) -I.
LDFLAGS  = -Llibnn -lnn -Ljsmn/wrapper -ljsmn -Llibcc -lcc -ldl -lpthread -lm
CCC      = gcc

all: $(TARGET)

$(TARGET): $(OBJECTS) libnn jsmn libcc
	$(CCC) $(OPT) $(OBJECTS) -o $@ $(LDFLAGS)

.PHONY: libnn jsmn libcc

libnn:
	$(MAKE) -C libnn

jsmn:
	$(MAKE) -C jsmn/wrapper

libcc:
	$(MAKE) -C libcc

clean:
	rm -f $(OBJECTS) *~ \#*\# $(TARGET)
	$(MAKE) -C libnn clean
	$(MAKE) -C jsmn/wrapper clean
	$(MAKE) -C libcc clean
	rm libnn jsmn libcc pcg-c-basic

$(OBJECTS): $(HFILES)
