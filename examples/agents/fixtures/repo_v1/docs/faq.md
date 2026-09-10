# FAQ

## How does ranking work?
We score every carrier that serves the lane on price, historical on-time rate and current
capacity, then return the top ten with an explanation for each.

## How fresh is carrier capacity?
Capacity is refreshed every fifteen minutes from carrier APIs, and immediately when a carrier
posts a truck.

## Can I override a ranking?
Yes. A dispatcher can pin or exclude a carrier per lane, and the override survives replanning.

## What happens if a carrier API is down?
The carrier is ranked from its last known capacity, and the response marks it as stale.

## Do you support partial loads?
Not yet. Every planned load is treated as a full truckload.
