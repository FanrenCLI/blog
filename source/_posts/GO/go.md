

```go
package main

import "fmt"

type Writer interface {
    Write([]byte) (int, error)
}

type StringWriter struct {
    str string
}

func (sw *StringWriter) Write(data []byte) (int, error) {
    sw.str += string(data)
    return len(data), nil
}

func main() {
    var w Writer = &StringWriter{}
    sw := w.(*StringWriter)
    sw.str = "Hello, World"
    fmt.Println(sw.str)
}
```