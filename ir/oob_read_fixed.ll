@.str = private constant [4 x i8] c"%d\0A\00"
declare i32 @printf(i8*, ...)

define i32 @main() {
entry:
  %arr = alloca [8 x i32], align 16
  %idx = alloca i32, align 4
  store i32 10, i32* %idx, align 4
  %0 = load i32, i32* %idx, align 4
  %cmp1 = icmp sge i32 %0, 0
  br i1 %cmp1, label %check2, label %if.end

check2:
  %cmp2 = icmp slt i32 %0, 8
  br i1 %cmp2, label %if.then, label %if.end

if.then:
  %arrayidx = getelementptr [8 x i32], [8 x i32]* %arr, i32 0, i32 %0
  %1 = load i32, i32* %arrayidx, align 4
  %call = call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 %1)
  br label %if.end

if.end:
  ret i32 0
}
