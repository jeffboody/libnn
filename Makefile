TARGET  = libnn.a
CLASSES = \
	nn_arch           \
	nn_batchNormLayer \
	nn_convLayer      \
	nn_coderLayer     \
	nn_dim            \
	nn_factLayer      \
	nn_flattenLayer   \
	nn_layer          \
	nn_loss           \
	nn_poolingLayer   \
	nn_skipLayer      \
	nn_tensor         \
	nn_weightLayer
SOURCE  = $(CLASSES:%=%.c)
OBJECTS = $(SOURCE:.c=.o)
HFILES  = $(CLASSES:%=%.h)
OPT     = -O2 -Wall
CFLAGS  = $(OPT)
ifeq ($(NN_USE_COMPUTE),1)
	# NN_USE_COMPUTE must also be included in CFLAGS of app in
	# order for header files to be included properly
	CFLAGS += -DNN_USE_COMPUTE
endif
ifeq ($(NN_GC_DEBUG),1)
	CFLAGS += -DNN_GC_DEBUG
endif
LDFLAGS = -lm
AR      = ar

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(AR) rcs $@ $(OBJECTS)

clean:
	rm -f $(OBJECTS) *~ \#*\# $(TARGET)

$(OBJECTS): $(HFILES)
