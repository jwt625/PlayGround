package main

import (
	"fmt"
	"github.com/robotn/gohook"
)

func main() {
	fmt.Println("Testing keyboard hook...")
	fmt.Println("Type something and press Ctrl+C to exit")
	
	hook.Register(hook.KeyDown, []string{}, func(e hook.Event) {
		fmt.Printf("Key pressed: %s (keycode: %d)\n", string(e.Keychar), e.Keycode)
	})

	s := hook.Start()
	<-s
}