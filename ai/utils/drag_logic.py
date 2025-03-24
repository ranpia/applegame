def drag_apples(positions, driver):
    if not positions:
        print("⚠️ 드래그할 사과 그룹 없음")
        return

    # 드래그할 사과 그룹 좌표
    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]
    x1, y1 = min(x_coords) - 15, min(y_coords) - 15
    x2, y2 = max(x_coords) + 15, max(y_coords) + 15

    print(f"🟢 드래그 실행: ({x1}, {y1}) → ({x2}, {y2})")  # 좌표 출력

    # 드래그 실행 (JavaScript 코드로)
    driver.execute_script(f"""
        const canvas = document.getElementById('canvas');
        const rect = canvas.getBoundingClientRect();
        const startX = rect.left + {x1};
        const startY = rect.top + {y1};
        const endX = rect.left + {x2};
        const endY = rect.top + {y2};

        const down = new MouseEvent('mousedown', {{
            bubbles: true, cancelable: true, view: window,
            clientX: startX, clientY: startY
        }});
        const move = new MouseEvent('mousemove', {{
            bubbles: true, cancelable: true, view: window,
            clientX: endX, clientY: endY
        }});
        const up = new MouseEvent('mouseup', {{
            bubbles: true, cancelable: true, view: window,
            clientX: endX, clientY: endY
        }});

        canvas.dispatchEvent(down);
        canvas.dispatchEvent(move);
        canvas.dispatchEvent(up);
    """)
