declare i8* @malloc(i64)
declare void @free(i8*)

define i32 @main() {
entry:
  %p = alloca i32*, align 8
  %call = call i8* @malloc(i64 4)
  %0 = bitcast i8* %call to i32*
  store i32* %0, i32** %p, align 8
  %1 = load i32*, i32** %p, align 8
  store i32 42, i32* %1, align 4
  %2 = load i32*, i32** %p, align 8
  %cast1 = bitcast i32* %2 to i8*
  call void @free(i8* %cast1)
  %3 = load i32*, i32** %p, align 8
  %cast2 = bitcast i32* %3 to i8*
  call void @free(i8* %cast2)
  ret i32 0
}
