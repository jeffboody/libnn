TARGET  = libnn.a
CLASSES = \
	nn_arch           \
	nn_batchNormLayer \
	nn_convLayer      \
	nn_dim            \
	nn_factLayer      \
	nn_flattenLayer   \
	nn_layer          \
	nn_loss           \
	nn_poolingLayer   \
	nn_tensor         \
	nn_weightLayer
SOURCE  = $(CLASSES:%=%.c)
OBJECTS = $(SOURCE:.c=.o)
HFILES  = $(CLASSES:%=%.h)
OPT     = -O2 -Wall
CFLAGS  = $(OPT)
LDFLAGS = -lm
AR      = ar

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(AR) rcs $@ $(OBJECTS)

clean:
	rm -f $(OBJECTS) *~ \#*\# $(TARGET)

$(OBJECTS): $(HFILES)
