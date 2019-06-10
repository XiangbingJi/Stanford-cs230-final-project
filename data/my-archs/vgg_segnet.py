
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    #input
    #to_input( '../examples/fcn8s/cats.jpg' ),

    #to_Conv
    to_Conv("conv1", 320, 192, 64, offset="(0,0,0)", to="(0,0,0)", height=44, depth=44, width=2 ),
    to_Conv("conv2", 320, 192, 64, offset="(1,0,0)", height=44, depth=44, width=2 ),
    to_Pool(name="pool1", caption="MAX POOL", offset="(2,0,0)", width=2, height=44, depth=44, opacity=0.5),

    to_Conv("conv3", 160, 96, 128, offset="(6,0,0)", height=32, depth=32, width=3 ),
    to_connection("pool1", "conv3"),
    to_Conv("conv4", 160, 96, 128, offset="(7,0,0)", height=32, depth=32, width=3 ),
    to_Pool(name="pool2", caption="MAX POOL",offset="(8,0,0)", width=3, height=32, depth=32, opacity=0.5),

    to_Conv("conv5", 80, 48, 256, offset="(12,0,0)", height=25, depth=25, width=4 ),
    to_connection("pool2", "conv5"),
    to_Conv("conv6", 80, 48, 256, offset="(13,0,0)", height=25, depth=25, width=4 ),
    to_Conv("conv7", 80, 48, 256, offset="(14,0,0)", height=25, depth=25, width=4 ),
    to_Pool(name="pool3", caption="MAX POOL", offset="(15,0,0)", width=4, height=25, depth=25, opacity=0.5),

    to_Conv("conv8", 40, 24, 512, offset="(18,0,0)", height=16, depth=16, width=5 ),
    to_connection("pool3", "conv8"),
    to_Conv("conv9", 40, 24, 512, offset="(19,0,0)", height=16, depth=16, width=5 ),
    to_Conv("conv10", 40, 24, 512, offset="(20,0,0)", height=16, depth=16, width=5 ),
    to_Pool(name="pool4", caption="MAX POOL", offset="(21,0,0)", width=5, height=16, depth=16, opacity=0.5),

    to_Conv("conv11", 20, 12, 512, offset="(24,0,0)", height=12, depth=12, width=5 ),
    to_connection("pool4", "conv11"),
    to_Conv("conv12", 20, 12, 512, offset="(25,0,0)", height=12, depth=12, width=5 ),
    to_Conv("conv13", 20, 12, 512, offset="(26,0,0)", height=12, depth=12, width=5 ),
    to_Pool(name="pool5", caption="MAX POOL", offset="(27,0,0)", width=5, height=12, depth=12, opacity=0.5),

    to_Norm(name="zero-pad-1", caption="ZERO PADDING", offset="(32,0,0)", height=12, depth=12, width=5 ),
    to_connection("pool5", "zero-pad-1"),
    to_Conv("conv-r-1", 20, 12, 512, offset="(33,0,0)", height=12, depth=12, width=5 ),
    to_Norm(name="batch-norm-1", caption="BATCH NORM", offset="(34,0,0)", height=12, depth=12, width=5 ),

    to_UnPool(name="unpool-1", caption="", offset="(37,0,0)", width=4, height=16, depth=16, opacity=0.5),
    to_Norm(name="zero-pad-2", caption="ZERO PADDING", offset="(38,0,0)", height=16, depth=16, width=4 ),
    to_connection("batch-norm-1", "unpool-1"),
    to_Conv("conv-r-2", 40, 24, 256, offset="(39,0,0)", height=16, depth=16, width=4 ),
    to_Norm(name="batch-norm-2", caption="BATCH NORM", offset="(40,0,0)", height=16, depth=16, width=4 ),

    to_UnPool(name="unpool-2", caption="", offset="(43,0,0)", width=3, height=25, depth=25, opacity=0.5),
    to_Norm(name="zero-pad-3", caption="ZERO PADDING", offset="(44,0,0)", height=25, depth=25, width=3 ),
    to_connection("batch-norm-2", "unpool-2"),
    to_Conv("conv-r-3", 80, 48, 128, offset="(45,0,0)", height=25, depth=25, width=3 ),
    to_Norm(name="batch-norm-3", caption="BATCH NORM", offset="(46,0,0)", height=25, depth=25, width=3 ),

    to_UnPool(name="unpool-3", caption="", offset="(49,0,0)", width=3, height=32, depth=32, opacity=0.5),
    to_Norm(name="zero-pad-4", caption="ZERO PADDING", offset="(50,0,0)", height=32, depth=32, width=3 ),
    to_connection("batch-norm-3", "unpool-3"),
    to_Conv("conv-r-4", 160, 96, 128, offset="(51,0,0)", height=32, depth=32, width=3 ),
    to_Norm(name="batch-norm-4", caption="BATCH NORM", offset="(52,0,0)", height=32, depth=32, width=3 ),

    to_UnPool(name="unpool-4", caption="", offset="(56,0,0)", width=2, height=44, depth=44, opacity=0.5),
    to_Norm(name="zero-pad-5", caption="ZERO PADDING", offset="(57,0,0)", height=44, depth=44, width=2 ),
    to_connection("batch-norm-4", "unpool-4"),
    to_Conv("conv-r-5", 320, 192, 64, offset="(58,0,0)", height=44, depth=44, width=2 ),
    to_Norm(name="batch-norm-5", caption="BATCH NORM", offset="(59,0,0)", height=44, depth=44, width=2 ),

    #to_SoftMax("soft1", 2 ,"(55,0,0)", caption="SOFTMAX	"  ),
    to_ConvSoftMax( name="soft1", s_filer=2, offset="(63,0,0)", width=2, height=44, depth=44, caption="SOFTMAX" ),
    to_connection("batch-norm-5", "soft1"),
    #block-001
    #to_ConvConvRelu( name='ccr_b1', s_filer=500, n_filer=(64,64), offset="(0,0,0)", to="(0,0,0)", width=(3,3), height=40, depth=40  ),
    #to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", width=1, height=32, depth=32, opacity=0.5),
    
    #*block_2ConvPool( name='b2', botton='pool_b1', top='pool_b2', s_filer=256, n_filer=128, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ),
    #*block_2ConvPool( name='b3', botton='pool_b2', top='pool_b3', s_filer=128, n_filer=256, offset="(2,0,0)", size=(25,25,4.5), opacity=0.5 ),
    #*block_2ConvPool( name='b4', botton='pool_b3', top='pool_b4', s_filer=64,  n_filer=512, offset="(3,0,0)", size=(16,16,5.5), opacity=0.5 ),

    #Bottleneck
    #block-005
    #to_ConvConvRelu( name='ccr_b5', s_filer=32, n_filer=(1024,1024), offset="(2,0,0)", to="(pool_b4-east)", width=(8,8), height=8, depth=8, caption="Bottleneck"  ),
    #to_connection( "pool_b4", "ccr_b5"),

    #Decoder
    #*block_Unconv( name="b6", botton="ccr_b5", top='end_b6', s_filer=64,  n_filer=512, offset="(2.1,0,0)", size=(16,16,5.0), opacity=0.5 ),
    #to_skip( of='ccr_b4', to='ccr_res_b6', pos=1.25),
    #*block_Unconv( name="b7", botton="end_b6", top='end_b7', s_filer=128, n_filer=256, offset="(2.1,0,0)", size=(25,25,4.5), opacity=0.5 ),
    #to_skip( of='ccr_b3', to='ccr_res_b7', pos=1.25),    
    #*block_Unconv( name="b8", botton="end_b7", top='end_b8', s_filer=256, n_filer=128, offset="(2.1,0,0)", size=(32,32,3.5), opacity=0.5 ),
    #to_skip( of='ccr_b2', to='ccr_res_b8', pos=1.25),    
    
    #*block_Unconv( name="b9", botton="end_b8", top='end_b9', s_filer=512, n_filer=64,  offset="(2.1,0,0)", size=(40,40,2.5), opacity=0.5 ),
    #to_skip( of='ccr_b1', to='ccr_res_b9', pos=1.25),
    
    #to_ConvSoftMax( name="soft1", s_filer=512, offset="(0.75,0,0)", to="(end_b9-east)", width=1, height=40, depth=40, caption="SOFT" ),
    #to_connection( "end_b9", "soft1"),
     
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    
