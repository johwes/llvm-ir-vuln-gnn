@.str = private constant [4 x i8] c"%d\0A\00"
declare i32 @printf(i8*, ...)

define i32 @main() {
entry:
  %arr = alloca [8 x i32], align 16
  %idx = alloca i32, align 4
  store i32 10, i32* %idx, align 4
  %0 = load i32, i32* %idx, align 4
  %arrayidx = getelementptr [8 x i32], [8 x i32]* %arr, i32 0, i32 %0
  %1 = load i32, i32* %arrayidx, align 4
  %call = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 %1)
  ret i32 0
}
