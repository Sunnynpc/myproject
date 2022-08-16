# # 计算器，控制台读取
s = input("请输入操作符")
#删除一行 ctrl + y
i = int( input( "请输入第一个操作数" ) )
#复制当前行 ctrl + d
#断点
j = int( input( "请输入第二个操作数" ) )
# if s == "+":
#     print( "%d + %d = %d"%( i,j,i+j ) )
# elif s == "-":
#     print( "%d - %d = %d"%( i,j,i-j ) )
# elif s == "*":
#     print("%d * %d = %d" % (i, j, i * j))
# elif s == "/":
#     print( "%d / %d = %s"%( i,j,i / j ) )
# else:
#     print( "输入错误" )
# #8步
d = { "+":"%d + %d = %d"%( i,j,i + j ),"-":"%d - %d = %d"%( i,j,i - j ),"*":"%d * %d = %d" % (i, j, i * j),"/":"%d / %d = %s"%( i,j,i / j ) }
print( d[s] )
#5步