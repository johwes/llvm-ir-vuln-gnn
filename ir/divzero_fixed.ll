declare i32 @printf(i8*, ...)
declare void @exit(i32)
declare i32 @fprintf(i8*, i8*, ...)
@.str = private constant [4 x i8] c"%d\0A\00"
@.err = private constant [17 x i8] c"divide by zero\0A\00"

define i32 @divide(i32 %a, i32 %b) {
entry:
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %zero, label %ok

zero:
  call void @exit(i32 1)
  unreachable

ok:
  %result = sdiv i32 %a, %b
  ret i32 %result
}

define i32 @main() {
entry:
  %call = call i32 @divide(i32 10, i32 0)
  %pf = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 %call)
  ret i32 0
}
