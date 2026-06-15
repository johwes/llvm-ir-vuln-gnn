declare i32 @printf(i8*, ...)
@.str = private constant [4 x i8] c"%d\0A\00"

define i32 @main() {
entry:
  %x = alloca i32, align 4
  store i32 0, i32* %x, align 4
  %0 = load i32, i32* %x, align 4
  %call = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 %0)
  ret i32 0
}
