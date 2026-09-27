identification division.
program-id. reference-sum.

data division.
working-storage section.
01 ws-count          pic 9(3) comp.
01 ws-index          pic 9(3) comp.
01 ws-value          pic s9(7) comp-5.
01 ws-total          pic s9(12) comp-3.
01 ws-display-total  pic -(12)9.

procedure division.
    accept ws-count

    if ws-count > 100
        display "ERROR:COUNT"
        stop run
    end-if

    move 0 to ws-total

    perform varying ws-index from 1 by 1 until ws-index > ws-count
        accept ws-value
        add ws-value to ws-total
    end-perform

    move ws-total to ws-display-total
    display "SUM=" ws-display-total
    stop run.