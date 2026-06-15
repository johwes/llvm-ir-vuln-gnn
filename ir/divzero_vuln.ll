declare i32 @printf(i8*, ...)
@.str = private constant [4 x i8] c"%d\0A\00"

define i32 @divide(i32 %a, i32 %b) {
entry:
  %result = sdiv i32 %a, %b
  ret i32 %result
}

define i32 @main() {
entry:
  %call = call i32 @divide(i32 10, i32 0)
  %pf = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 %call)
  ret i32 0
}
