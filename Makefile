TARGET  = libnn.a
CLASSES = \
	nn_arch             \
	nn_batchNormLayer   \
	nn_convLayer        \
	nn_coderLayer       \
	nn_dim              \
	nn_encdecLayer      \
	nn_engine           \
	nn_factLayer        \
	nn_lanczosLayer     \
	nn_lanczosResampler \
	nn_layer            \
	nn_loss             \
	nn_resLayer         \
	nn_reshapeLayer     \
	nn_skipLayer        \
	nn_tensorStats      \
	nn_tensor           \
	nn_urrdbBlockLayer  \
	nn_urrdbLayer       \
	nn_urrdbNodeLayer   \
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
